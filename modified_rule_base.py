import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from rule_generation import rule_base

class rule_base_modification(rule_base):

    def remove_duplicate(self):
        
        raw_rule_base = []
        for i in range(40):
            raw_rule_base.append(self.rule_generate(i))

        
        final_rule_base = []
        rule_dict = {}

        for rule in raw_rule_base:
            rule_key = (rule[0], rule[1]) 
            current_label = rule[2]
            current_weight = rule[3]
            
            if rule_key in rule_dict:
                rule_dict[rule_key].append([current_label, current_weight])
            else:
                rule_dict[rule_key] = [[current_label, current_weight]]

        
        for i in range(3):
            for j in range(3):
                if (i, j) in rule_dict:
                    
                    
                    total_weights_per_class = {}
            
                    for label, weight in rule_dict[(i, j)]:
                        if label not in total_weights_per_class:
                            total_weights_per_class[label] = 0
                        
                        
                        total_weights_per_class[label] += weight
                    
                    
                    if total_weights_per_class:
                        winning_class = max(total_weights_per_class, key=total_weights_per_class.get)
                        
                        
                        winning_weight = total_weights_per_class[winning_class] 
                    
                        final_rule_base.append([i, j, winning_class, winning_weight])

        print(f"Final rule count: {len(final_rule_base)}")

     
        membership = ["Low", "Medium", "High"]
        class_label = ["Healthy", "Mild", "Severe"]
        for rule in final_rule_base:
         
             print(f"IF rms_g is {membership[rule[0]]} AND kurtosis is {membership[rule[1]]} THEN class is {class_label[rule[2]]}")

  
        
    
rule_modofier = rule_base_modification()
rule_modofier.remove_duplicate()