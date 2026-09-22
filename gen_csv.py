import pandas as pd
import os
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="close_loop",
                        help="Directory containing the closed-loop evaluation summaries")
    parser.add_argument("--out_dir", type=str, default="close_loop/csv",
                        help="Directory to write the generated CSV files")
    args = parser.parse_args()
    results_dir = args.results_dir
    # task_list = ["maniskill","dubins","cargoal"]
    task_list = ["maniskill_0.0","maniskill_0.2"]
    type_list = ["dynamics","critic","pid"]
    csv_dir = args.out_dir
    csv_header = ["type","finetune","backbone","avg_steps","avg_vio","avg_min_hj","avg_success_rate","avg_reward","avg_interventions","avg_intervention_rate","avg_encode_time","avg_dynamics_time","avg_critic_time","avg_actor_time"]
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    for task in task_list:
        csv_path = f"{csv_dir}/{task}.csv"
        csv = pd.DataFrame(columns=csv_header)
        for type in type_list:
            if type == "pid":
                summary_path = f"{results_dir}/{task}/{type}/summary.txt"
                with open(summary_path, "r") as f:
                    summary = f.readlines()
                avg_steps = float(summary[2].strip().split(": ")[1])
                avg_violations = float(summary[3].strip().split(": ")[1])
                avg_min_hj = float(summary[4].strip().split(": ")[1])
                avg_success_rate = float(summary[5].strip().split(": ")[1])
                avg_reward = float(summary[6].strip().split(": ")[1])
                csv.loc[len(csv)] = [type,False,None,avg_steps,avg_violations,avg_min_hj,avg_success_rate,avg_reward,None,None,None,None,None,None]
            elif type == "critic":
                backbone_list = ["dino","dino_cls","vc1","r3m","resnet","scratch","full_scratch","dino_ft","dino_cls_ft","vc1_ft","r3m_ft","resnet_ft","scratch_ft"]
                for backbone in backbone_list:
                    summary_path = f"{results_dir}/{task}/{type}/{backbone}/summary.txt"
                    with open(summary_path, "r") as f:
                        summary = f.readlines()
                    avg_steps = float(summary[2].strip().split(": ")[1])
                    avg_violations = float(summary[3].strip().split(": ")[1])
                    avg_min_hj = float(summary[4].strip().split(": ")[1])
                    avg_success_rate = float(summary[5].strip().split(": ")[1])
                    avg_reward = float(summary[6].strip().split(": ")[1])
                    intervention = summary[7].strip().split(": ")[1].strip().split(" ")
                    avg_interventions = float(intervention[0])
                    avg_intervention_rate = float(intervention[1][:-2][1:])/100
                    if "ft" in backbone:
                        finetune = True
                    else:
                        finetune = False
                    csv.loc[len(csv)] = [type,finetune,backbone,avg_steps,avg_violations,avg_min_hj,avg_success_rate,avg_reward,avg_interventions,avg_intervention_rate,None,None,None,None]
            elif type == "dynamics":
                backbone_list = ["dino","dino_cls","vc1","r3m","resnet","scratch"]
                for backbone in backbone_list:
                    summary_path = f"{results_dir}/{task}/{type}/{backbone}/summary.txt"
                    with open(summary_path, "r") as f:
                        summary = f.readlines()
                    avg_steps = float(summary[2].strip().split(": ")[1])
                    avg_violations = float(summary[3].strip().split(": ")[1])
                    avg_min_hj = float(summary[4].strip().split(": ")[1])
                    avg_success_rate = float(summary[5].strip().split(": ")[1])
                    avg_reward = float(summary[6].strip().split(": ")[1])
                    intervention = summary[7].strip().split(": ")[1].strip().split(" ")
                    avg_interventions = float(intervention[0])
                    avg_intervention_rate = float(intervention[1][:-2][1:])/100
                    finetune = False
                    avg_encode_time = float(summary[8].strip().split(": ")[1][:-1])/100
                    avg_dynamics_time = float(summary[9].strip().split(": ")[1][:-1])/100
                    avg_critic_time = float(summary[10].strip().split(": ")[1][:-1])/100
                    if "dubins" in task:
                        avg_actor_time = None
                    else:
                        avg_actor_time = float(summary[11].strip().split(": ")[1][:-1])/100
                    csv.loc[len(csv)] = [type,finetune,backbone,avg_steps,avg_violations,avg_min_hj,avg_success_rate,avg_reward,avg_interventions,avg_intervention_rate,avg_encode_time,avg_dynamics_time,avg_critic_time,avg_actor_time]
        csv.to_csv(csv_path, index=False)
