# Development of 5G based UAV Augmented Intelligent Monitoring and Surveillance System.


## Discription

The project titled “Development of 5G based UAV Augmented Intelligent Monitoring and Surveillance System”,
sanctioned by MeitY, aims at utilizing the benefits of 5G network’s approachability, service architecture and speed
with IoT and Machine Learning to roll out citizen centric monitoring and security applications. The service
architecture of 5G communication technology will lend support to parallel execution of various M2M services. To
offer more accurate and reliable monitoring and security, data is collected using policy compliant UAVs, and works
in augmentation to sensors’ data. The proposed work will implement a layered framework for seamless D2D
communication and control while collaborating with modern technologies over 5G communication networks.
Different layers will support varied functionality and will offer choice of services to end users. The integration of
data storage on Cloud with Machine Learning models will enable real-time availability of data and help to run
independent data analytics and applications offering flexible services to all stakeholders. The proposed framework
can be used for deployment of applications like Mob behavior analysis, Traffic analysis, Detection of intruders,
Verification of land records, Illegal sand mining, Monitoring and detection of breeding areas for various diseases,
etc.
Currently, we are working with UAVs to surveil mob behavior and detect violence and fight scenes. Our drone has a
Raspberry Pi augmented with it which works on 4G/5G communication network. Drone captures real time aerial
videos using the camera mounted over Raspberry Pi and sends the captured data (videos) as RTP packets over a
UDP channel to our deep learning model residing on cloud server. The model takes the data, processes it and
identifies the activities going on in the video so as to detect fight scenes and violence. Once an anomaly is detected,
alerts are generated and sent to the stakeholders such as Police or other authorities.
Our model has a number of advantages that are mentioned as follows:
1. For transmitting the videos from drone to the cloud server/ DL model, ground control station is not required as
an intermediary because the videos are captured by the camera which is mounted over the Raspberry Pi.
Raspberry Pi works on 4G/5G network which collects data and sends it directly to the cloud server. This makes
the transmission faster and efficient.
2. Integration of a Raspberry Pi or any other computing device with drones is not possible for end users otherwise
because they are not authorized to do so. We can provide our users with a UAV that already has a computing
device augmented with it making it more powerful and efficient.
3. Transmission of videos over the network is seamless and secure with all the capabilities of data encryption and
security from 4G/LTE.
4. Drone manufacturing companies can create their own cloud server and integrate our deep learning model with
it, which provides their customers with an automated system for the identification of fight scenes, fires and
other disasters.
## Demo Flight Video
![](https://github.com/telelabchd/finalrelease-uav-telelab-meity/blob/main/ezgif.com-gif-maker.gif)

## Demo Social Distancing
![](https://github.com/telelabchd/finalrelease-uav-telelab-meity/blob/main/sample%20distancing.gif)

## Demo video various applications
![](https://github.com/telelabchd/finalrelease-uav-telelab-meity/blob/main/ezgif.com-gif-maker%20(1)%20(2).gif)
