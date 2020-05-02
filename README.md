# Diplom
Work with population density model

This project devoted to population dynamics with growing carrying capacity. This model describes next equation

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;u}{\partial&space;t}=D\frac{\partial^2&space;u}{\partial&space;x^2}&space;&plus;&space;au(1-\frac{u}{T})&space;-&space;σu" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;u}{\partial&space;t}=D\frac{\partial^2&space;u}{\partial&space;x^2}&space;&plus;&space;au(1-\frac{u}{T})&space;-&space;σu" title="\frac{\partial u}{\partial t}=D\frac{\partial^2 u}{\partial x^2} + au(1-\frac{u}{T}) - σu" /></a>

This equation resolved thomasAlgorithm or Tridiagonal matrix algorithm, in  3 different variant for realization of carrying capacity:
* when K = constant (k_constant)
* when K = Time-dependent function (k_function)
* and case global consumption of resources, where K have description by integrall (k_integrall)
