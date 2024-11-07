# SMC-Searcher:

## Overview:
This work studies a special multi-robot search problem, namely multi-robot adversarial search (MuRAS) problem, where the moving target acts to avoid being detected by the robot team. We get inspirations from the traffic light signal for multi-vehicle coordinated navigation, and invent a Signal Mediated Coordination (SMC) method to coordinate the decentralized decision-making process of the searching robots. In addition to the normal multi-robot search strategy, which directs the robots towards a high-rewarding path of detecting the moving target, SMC imposes a global coordination signal, which enforces that (1) different robots react to the same coordination signal differently and (2) the same robot reacts to different signals differently. In this way, the multiple searching robots will collectively exhibit an exogenously diversified while endogenously coordinated search strategy, which is difficult for the adversarial moving target to exploit and thereby avoid being captured.
The framework of SMC-Searcher is as follows:
![Framework](https://github.com/user-attachments/assets/c5ad5d4d-35e0-4a14-a714-00cb8cf45b96)
