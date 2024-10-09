# LNO-for-Van-der-Pol-Oscillations
Discusses the application of the Laplace Neural Operator in the context of the forced Van der Pol oscillator use case. In the Van der Pol oscillator use case, the primary objective is to learn a neural operator Gθ that maps the external forcing function f(t) to the system's response x(t). 

## Requirements
Check the 'requirements.txt' file within the repo.

## Data
The dataset employed in this project is in .mat format. To generate the samples to train LNO, I considered a sinusoidal forcing function, ftrain(t) = Asin(5t), where the amplitude A ∈ [0.05, 10] with an interval δA = 0.05, therefore Ntrain = 200. Each sample is discretized into 2048 temporal points and the time interval is ∆t = 0.01 seconds. The response is calculated by a versatile ODE solver—ode45 on MATLAB. For validating and testing the neural operator, I generated, the same way, Nvali = 50 samples and Ntest = 130 samples.

The dataset used in this project is too large to be hosted on GitHub. However, if you are interested in accessing the dataset for research, testing, or development purposes, please feel free to contact me.

## Connect with Me
- LinkedIn: [ErikJon Pérez](https://www.linkedin.com/in/erikjon-perez-mardaras/)
- GitHub: [Github](https://github.com/erikjonperez)

## Acknowledgements
I would like to thank qianyingcao for their original work, which this project is based on. You can find their repository here: [Original Repo](https://github.com/qianyingcao/Laplace-Neural-Operator).

This project uses code licensed under the MIT License.
