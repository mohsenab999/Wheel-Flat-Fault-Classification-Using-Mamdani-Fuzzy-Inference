import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from modified_rule_base import rule_base_modification
from rule_generation import rule_base
class mamdani_inference(rule_base):
    
   

    def rule_1_antecedent(self, input):

        x_1_fire = self.triangular_membership(input[0], self.memship_list[0]["Low"]["m"], self.memship_list[0]["Low"]["b"])
        x_2_fire = self.triangular_membership(input[1], self.memship_list[1]["Low"]["m"], self.memship_list[1]["Low"]["b"])
        return [(x_1_fire * x_2_fire), "Healthy"]
    
    def rule_2_antecedent(self, input):

        
        x_1_fire = self.triangular_membership(input[0], self.memship_list[0]["Low"]["m"], self.memship_list[0]["Low"]["b"])
        x_2_fire = self.triangular_membership(input[1], self.memship_list[1]["Medium"]["m"], self.memship_list[1]["Medium"]["b"])
        return [(x_1_fire * x_2_fire), "Mild"]
        
    def rule_3_antecedent(self, input):

        x_1_fire = self.triangular_membership(input[0], self.memship_list[0]["Low"]["m"], self.memship_list[0]["Low"]["b"])
        x_2_fire = self.triangular_membership(input[1], self.memship_list[1]["High"]["m"], self.memship_list[1]["High"]["b"])
        return [(x_1_fire * x_2_fire), "Severe"]

    def rule_4_antecedent(self, input):

        x_1_fire = self.triangular_membership(input[0], self.memship_list[0]["Medium"]["m"], self.memship_list[0]["Medium"]["b"])
        x_2_fire = self.triangular_membership(input[1], self.memship_list[1]["Low"]["m"], self.memship_list[1]["Low"]["b"])
        return [(x_1_fire * x_2_fire), "Mild"]

    
    def rule_5_antecedent(self, input):

        x_1_fire = self.triangular_membership(input[0], self.memship_list[0]["Medium"]["m"], self.memship_list[0]["Medium"]["b"])
        x_2_fire = self.triangular_membership(input[1], self.memship_list[1]["Medium"]["m"], self.memship_list[1]["Medium"]["b"])
        return [(x_1_fire * x_2_fire), "Mild"]

    def rule_6_antecedent(self, input):

        x_1_fire = self.triangular_membership(input[0], self.memship_list[0]["Medium"]["m"], self.memship_list[0]["Medium"]["b"])
        x_2_fire = self.triangular_membership(input[1], self.memship_list[1]["High"]["m"], self.memship_list[1]["High"]["b"])
        return [(x_1_fire * x_2_fire), "Severe"]

    def rule_7_antecedent(self, input):

        x_1_fire = self.triangular_membership(input[0], self.memship_list[0]["High"]["m"], self.memship_list[0]["High"]["b"])
        x_2_fire = self.triangular_membership(input[1], self.memship_list[1]["Low"]["m"], self.memship_list[1]["Low"]["b"])
        return [(x_1_fire * x_2_fire), "Severe"]
    
    def rule_8_antecedent(self, input):

        x_1_fire = self.triangular_membership(input[0], self.memship_list[0]["High"]["m"], self.memship_list[0]["High"]["b"])
        x_2_fire = self.triangular_membership(input[1], self.memship_list[1]["Medium"]["m"], self.memship_list[1]["Medium"]["b"])
        return [(x_1_fire * x_2_fire), "Severe"]
    
    def rule_9_antecedent(self, input):

        x_1_fire = self.triangular_membership(input[0], self.memship_list[0]["High"]["m"], self.memship_list[0]["High"]["b"])
        x_2_fire = self.triangular_membership(input[1], self.memship_list[1]["High"]["m"], self.memship_list[1]["High"]["b"])
        return [(x_1_fire * x_2_fire), "Severe"]
    



    def classifier (self, input_data):

        rule_firing_list = [self.rule_1_antecedent, self.rule_2_antecedent, self.rule_3_antecedent, self.rule_4_antecedent, self.rule_5_antecedent, self.rule_6_antecedent, self.rule_7_antecedent, self.rule_8_antecedent, self.rule_9_antecedent]

        label_domain = np.linspace(0, 2, 500)



        aggregated_output = np.zeros_like(label_domain)

        for rule in rule_firing_list:
            rule_result = rule(input_data)
            base_shape = self.triangular_membership(label_domain, self.memship_list[2][rule_result[1]]["m"], self.memship_list[2][rule_result[1]]["b"])

            clipped_shape = base_shape * rule_result[0]

            aggregated_output = np.fmax(clipped_shape, aggregated_output)

        numerator = np.sum(label_domain * aggregated_output)
        denominator = np.sum(aggregated_output)
        if denominator == 0:
            final_label_value = 0.0  # Default value if no rules fire
        else:
            final_label_value = numerator / denominator
            
        label = int(round(final_label_value))

        label_list = ["Healthy", "Mild", "Severe"]
        print(f"{label_list[label]}")

