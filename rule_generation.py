import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class rule_base:

    DATA_DIR = Path(__file__).resolve().parent
    df = pd.read_csv(DATA_DIR / "WheelFlat_Dataset1_Train (2).csv")
    df.columns = df.columns.str.strip()

    y = df["kurtosis"].to_numpy()
    x = df["rms_g"].to_numpy()
    z = df["label"].to_numpy()

    memship_list = [

        {                                            
            "Low":       {"m": 0.385,    "b": 0.228}, 
            "Medium":    {"m": 0.613,    "b": 0.228},  
            "High":      {"m": 0.841,    "b": 0.228},  
        },

        {                                            
            "Low":       {"m": 3.9,  "b": 1.58}, 
            "Medium":    {"m": 5.48,  "b": 1.58},  
            "High":      {"m": 7.06,  "b": 1.58},  
        },

        {                                            
            "Healthy":     {"m": 0,  "b": 1}, 
            "Mild":        {"m": 1,  "b": 1},  
            "Severe":      {"m": 2,  "b": 1},  
        }
    ]   


    def triangular_membership(self,x_domain, m, b):

        t = np.maximum(1 - (np.abs(x_domain - m)/b), 0) 
        return t

    def rule_generate(self,i):

        x_label = ""
        y_label = ""
        t_x_v = []
        t_y_v = []
        t_z_v = []

        t_x_v.append(self.triangular_membership(self.x[i], self.memship_list[0]["Low"]["m"],    self.memship_list[0]["Low"]["b"]))
        t_x_v.append(self.triangular_membership(self.x[i], self.memship_list[0]["Medium"]["m"], self.memship_list[0]["Medium"]["b"]))
        t_x_v.append(self.triangular_membership(self.x[i], self.memship_list[0]["High"]["m"],   self.memship_list[0]["High"]["b"]))

        t_y_v.append(self.triangular_membership(self.y[i], self.memship_list[1]["Low"]["m"],    self.memship_list[1]["Low"]["b"]))
        t_y_v.append(self.triangular_membership(self.y[i], self.memship_list[1]["Medium"]["m"], self.memship_list[1]["Medium"]["b"]))
        t_y_v.append(self.triangular_membership(self.y[i], self.memship_list[1]["High"]["m"],   self.memship_list[1]["High"]["b"]))

        t_z_v.append(self.triangular_membership(self.z[i], self.memship_list[2]["Healthy"]["m"], self.memship_list[2]["Healthy"]["b"]))
        t_z_v.append(self.triangular_membership(self.z[i], self.memship_list[2]["Mild"]["m"],    self.memship_list[2]["Mild"]["b"]))
        t_z_v.append(self.triangular_membership(self.z[i], self.memship_list[2]["Severe"]["m"],  self.memship_list[2]["Severe"]["b"]))

        t_x_v_max = np.max(t_x_v)
        t_y_v_max = np.max(t_y_v)
        t_z_v_max = np.max(t_z_v)
        weight = t_x_v_max * t_y_v_max
        t_x_v_max_ind = t_x_v.index(t_x_v_max)
        t_y_v_max_ind = t_y_v.index(t_y_v_max)
        t_z_v_max_ind = t_z_v.index(t_z_v_max)

        return [t_x_v_max_ind, t_y_v_max_ind, t_z_v_max_ind, weight]



    
membership = ["Low", "Medium", "High"]
class_label = ["Healthy", "Mild", "Severe"]
Rule_base = rule_base()
for i in range(40):
        rule_parameters = Rule_base.rule_generate(i)
        