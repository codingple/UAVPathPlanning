# UAVPathPlanning
#### A path planning for UAV collecting sensor data using polynomial regression  
  
### Description  
This program is an implementation of the follwing paper:  
> Weighted Harvest-Then-Transmit: UAV-Enabled Wireless Powered Communication Networks  
> [10.1109/ACCESS.2018.2882128](https://ieeexplore.ieee.org/document/8540379)  
  
The purpose is to generate a optimum path for collecting sensor data. 
The polynomial regression using GPS coordinates of deployed sensors generates an equation used in the path.  
In addition, this program corrects the path to communicate with all sensors to collect.  
  
### Prerequisite  
- Python (ver. 3.6)  
- Tensorflow for python 3.6  
- Matplotlib  

### Description  
* **Initiation**  
&nbsp;  
<p align="center">
  <img src="./images/init1.png" width="90%" height="90%">
  <img src="./images/init2.png" width="90%" height="90%">
</p>  
&nbsp;  

  
* **Training**  
&nbsp;  
<p align="center">
  <img src="./images/training.png" width="90%" height="90%">
</p>  
&nbsp;  

  
* **Illustration**  
&nbsp;  
<p align="center">
  <img src="./images/Illust1.png" width="90%" height="90%">
  <img src="./images/illust2.png" width="90%" height="90%">
</p>  
&nbsp;  

  
### Results  
The following figure is an example of result:  
<p align="center">
  <img src="./images/result.png" width="50%" height="50%">
</p>  
&nbsp; 

In contrast, this is the result when is a general regression without using "range correction(1st Path)":
&nbsp;  
<p align="center">
  <img src="./images/regression.png" width="50%" height="50%">
</p>  
&nbsp; 
