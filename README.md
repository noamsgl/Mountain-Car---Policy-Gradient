# Solve Mountain Car using Policy Gradient
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/noamsgl/Mountain-Car---Policy-Gradient">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Reinforcement-Learning 2021, Homework 4</h3>

  <p align="center">
    A Policy Gradient solution to the MountainCar environment.
    <br />
  </p>
</p>



<!-- ABOUT THE PROJECT -->
## About The Project

This software was developed in the Reinforcement Learning 2021, Semester A, course at [Ben Gurion University](https://in.bgu.ac.il/en/pages/default.aspx).

It implements the Policy Gradient algorithm to solve the [MountainCar](https://gym.openai.com/envs/MountainCar-v0/) environment. The policy is parameterized via [radial basis function](https://en.wikipedia.org/wiki/Radial_basis_function).
If a thetas.csv file is found, it will first simulate a test drive using the weights.  
Finally, a call to plt.show() will display a plot of the empirical (monte-carlo estimated) value function over the training period.
### Prerequisites

* [Python](https://www.python.org/downloads/)
* Python libraries
  ```sh
  pip install numpy
  pip install gym
  pip install tqdm
  pip install matplotlib
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/noamsgl/Mountain-Car---Policy-Gradient.git
   ```

<!-- USAGE EXAMPLES -->
## Usage

```sh
python3 main.py
```
During simulation and training, you will see the rendered environment:
[![Product Name Screen Shot][product-render]](https://github.com/noamsgl/Mountain-Car---Policy-Gradient)

Finally, you will get a plot of the estimated value function as a function of time step:
[![Product Name Screen Shot][product-plot]](https://github.com/noamsgl/Mountain-Car---Policy-Gradient)




<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Noam Siegel - noamsi@post.bgu.ac.il

Dolev Orgad - dolevi101@gmail.com

Project Link: [https://github.com/noamsgl/Mountain-Car---Policy-Gradient](https://github.com/noamsgl/Mountain-Car---Policy-Gradient)


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-render]: images/render.png
[product-plot]: images/plot.png

