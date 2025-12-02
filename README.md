# Bus-Capacity-Prediction
## Purpose
Over the past year, the University of Florida cut millions of dollars from Regional Transit
System’s (RTS) annual budget, representing a twenty percent decrease. Additionally, more cuts
are consistently being proposed. RTS operates bus routes servicing students, faculty, and other
members of the Gainesville community (https://taps.ufl.edu/bus-route-changes/, https://www.gainesvillefl.gov/Government-Pages/Government/Departments/Transportation/RT
S/UF-Transit-Services-Agreement). Assuming that these cuts are permanent, they
present a dire need to optimize the bus system. When making decisions on which routes to cut,
consolidate, or shorten (as has been the case for many routes), it would be highly beneficial to
accurately predict the number of passengers on a bus at a given point, so as to prioritize the
highly trafficked routes. Unfortunately, bus drivers cannot feasibly be expected to always keep
track of their passenger count, as this would require counting both oncomers and ongoers.
Instead, drivers are responsible for checking the IDs of oncoming passengers and pressing a
button to count each one. 

## Task
Given the count of oncoming passengers, paired with the busses’ 
previous stop, the goal is to confidently predict the current number of passengers on the
bus. An accuracte prediction is defined as +/-5% of the known passenger count, and a target accuracy will be 95%.
A variety of machine learning frameworks and libraries will be tested. Predominantly, I am interested in experimenting with RNNs. 

## Inputs and Outputs
Input row: 
Previous stop (Index of stop along the bus's route), Previous estimated passengers (Before stopping at previous stop), Number of oncomers at previous stop
Output: 
Predicted number of passengers after stop

## Running
To train the model and generate the accuracy, run model.py after installing the requirements from requirements.txt
## Author
Christopher Tyler- https://www.linkedin.com/in/christopher-tyler-404941227/
