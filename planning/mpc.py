import torch
import hydra
import copy
import numpy as np
from einops import rearrange, repeat
from utils import slice_trajdict_with_t
from .base_planner import BasePlanner


class MPCPlanner(BasePlanner):
    """
    an online planner so feedback from env is allowed
    """

    def __init__(
        self,
        max_iter,
        n_taken_actions,
        sub_planner,
        wm,
        env,  # for online exec
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="mpc",
        log_filename="logs.json",
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.env = env
        self.max_iter = np.inf if max_iter is None else max_iter
        self.n_taken_actions = n_taken_actions
        self.logging_prefix = logging_prefix
        sub_planner["_target_"] = sub_planner["target"]
        self.sub_planner = hydra.utils.instantiate(
            sub_planner,
            wm=self.wm,
            action_dim=self.action_dim,
            objective_fn=self.objective_fn,
            preprocessor=self.preprocessor,
            evaluator=self.evaluator,  # evaluator is shared for mpc and sub_planner
            wandb_run=self.wandb_run,
            log_filename=None,
        )
        self.is_success = None
        self.action_len = None  # keep track of the step each traj reaches success
        self.iter = 0
        self.planned_actions = []

    def _apply_success_mask(self, actions):
        device = actions.device
        mask = torch.tensor(self.is_success).bool()
        actions[mask] = 0
        masked_actions = rearrange(
            actions[mask], "... (f d) -> ... f d", f=self.evaluator.frameskip
        )
        masked_actions = self.preprocessor.normalize_actions(masked_actions.cpu())
        masked_actions = rearrange(masked_actions, "... f d -> ... (f d)")
        actions[mask] = masked_actions.to(device)
        return actions

    def plan(self, obs_0, obs_g, actions=None):
        """
        actions is NOT used
        Returns:
            actions: (B, T, action_dim) torch.Tensor
        """
        n_evals = obs_0["visual"].shape[0]
        self.is_success = np.zeros(n_evals, dtype=bool)
        self.action_len = np.full(n_evals, np.inf)
        init_obs_0, init_state_0 = self.evaluator.get_init_cond()

        cur_obs_0 = obs_0
        memo_actions = None
        while not np.all(self.is_success) and self.iter < self.max_iter:
            self.sub_planner.logging_prefix = f"plan_{self.iter}"
            actions, _ = self.sub_planner.plan(
                obs_0=cur_obs_0,
                obs_g=obs_g,
                actions=memo_actions,
            )  # (b, t, act_dim)
            taken_actions = actions.detach()[:, : self.n_taken_actions]
            self._apply_success_mask(taken_actions)
            memo_actions = actions.detach()[:, self.n_taken_actions :]
            self.planned_actions.append(taken_actions)

            print(f"MPC iter {self.iter} Eval ------- ")
            action_so_far = torch.cat(self.planned_actions, dim=1)
            self.evaluator.assign_init_cond(
                obs_0=init_obs_0,
                state_0=init_state_0,
            )
            
            ''' i added this
            Before you run eval_actions(action_so_far, …), you must guarantee that the evaluator’s internal environment is reset to the very original initial condition. That way:
            The entire chain in action_so_far is played sequentially beginning from exactly the same start that the original obs_0 described.
            '''
            logs, successes, e_obses, e_states = self.evaluator.eval_actions(
                action_so_far,
                self.action_len,
                filename=f"plan{self.iter}",
                save_video=True,
            )
            new_successes = successes & ~self.is_success  # Identify new successes
            self.is_success = (
                self.is_success | successes
            )  # Update overall success status
            self.action_len[new_successes] = (
                (self.iter + 1) * self.n_taken_actions
            )  # Update only for the newly successful trajectories

            print("self.is_success: ", self.is_success)
            logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
            logs.update({"step": self.iter + 1})
            self.wandb_run.log(logs)
            self.dump_logs(logs)

            # update evaluator's init conditions with new env feedback
            e_final_obs = slice_trajdict_with_t(e_obses, start_idx=-1)
            cur_obs_0 = e_final_obs
            e_final_state = e_states[:, -1]
            '''
            from now on, start your next sub‐plan from this new e_final_state (and corresponding image e_final_obs).” 
            That’s how you get closed‐loop feedback: you re‐plan from the environment’s new actual state,
            rather than from some stale “predicted” state.
    
            note that if you look at output files you would see planX_n_failure/success.mp4 and each of these mp4 executes a sequence
            of actions starting from initial state up until the end. so each time X increases (1 mpc iteration extra) we get a longer video
            
            
            therefore:
            first assign_init_cond resets the evaluator to the very first initial condition so that eval_actions(action_so_far,…) is a rollout of the whole chain from t=0
            Second assign_init_cond sets the evaluator to the new current state t=k+1 * n_takenactions so that next iteration’s sub‐planner sees the correct “live” initial state for its optimization.
            ''' 
            self.evaluator.assign_init_cond(
                obs_0=e_final_obs,
                state_0=e_final_state,
            )
            self.iter += 1
            self.sub_planner.logging_prefix = f"plan_{self.iter}"



        planned_actions = torch.cat(self.planned_actions, dim=1)
        self.evaluator.assign_init_cond(
            obs_0=init_obs_0,
            state_0=init_state_0,
        )

        return planned_actions, self.action_len
