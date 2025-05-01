# Cosmic Ray Muon Trigger System Simulation

This is Cosmic Ray Muon Trigger System Simulation, a small project I got the chance to work on with my colleage and dear friend Georgios Koryfidis. The project was done in the context of the course "Graduate Studies in Experimental Data Analysis" and its goal is to simulate cosmic ray muons that pass through two square surfaces acting like detectors.

The top surface creates events (points) in a uniform way. The theta angle (or zenith angle) generation follows the cos^2 distribution, while the phi angle (or polar angle) generation is uniform. The track of a muon may or may not pass through the bottom surface (after its birth on the top surface) depending on the value of its zenith angle. If the angle is smaller or equal to the maximum value of theta then the track hits the bottom detector and we say we have a trigger event.

Download the code muons.py. Make sure you have all the nescassary packages installed. Run the code and enjoy!
