import numpy as np
import matplotlib.pyplot as plt


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



def triangular_membership(x_domain, m, b):

    t = np.maximum(1 - (np.abs(x_domain - m)/b), 0) 
    return t

x_1 = np.linspace(0.157, 1.072, 500)
x_2 = np.linspace(2.32, 8.64, 500)
z = np.linspace(0, 2, 500)

y1_low = triangular_membership(x_1, memship_list[0]["Low"]["m"], memship_list[0]["Low"]["b"])
y1_med = triangular_membership(x_1, memship_list[0]["Medium"]["m"], memship_list[0]["Medium"]["b"])
y1_high = triangular_membership(x_1, memship_list[0]["High"]["m"], memship_list[0]["High"]["b"])

y2_low = triangular_membership(x_2, memship_list[1]["Low"]["m"], memship_list[1]["Low"]["b"])
y2_med = triangular_membership(x_2, memship_list[1]["Medium"]["m"], memship_list[1]["Medium"]["b"])
y2_high = triangular_membership(x_2, memship_list[1]["High"]["m"], memship_list[1]["High"]["b"])

o_health = triangular_membership(z, memship_list[2]["Healthy"]["m"], memship_list[2]["Healthy"]["b"])
o_mild = triangular_membership(z, memship_list[2]["Mild"]["m"], memship_list[2]["Mild"]["b"])
o_severe = triangular_membership(z, memship_list[2]["Severe"]["m"], memship_list[2]["Severe"]["b"])

plt.plot(x_1, y1_low, color = "red", label = "LOW")
plt.plot(x_1, y1_med, color = "blue", label = "MEDIUM")
plt.plot(x_1, y1_high, color = "black", label = "HIGH")
plt.ylabel("Degree of Membership")
plt.xlabel("rms_g")
plt.legend()
plt.show()

plt.plot(x_2, y2_low, color = "red", label = "LOW")
plt.plot(x_2, y2_med, color = "blue", label = "MEDIUM")
plt.plot(x_2, y2_high, color = "black", label = "HIGH")
plt.ylabel("Degree of Membership")
plt.xlabel("kurtosis")
plt.legend()
plt.show()

plt.plot(z, o_health, color = "red", label = "HEALTHY")
plt.plot(z, o_mild,   color = "blue", label = "MILD")
plt.plot(z, o_severe, color = "black", label = "SEVERE")
plt.ylabel("Degree of Membership")
plt.xlabel("class label")
plt.legend()
plt.show()