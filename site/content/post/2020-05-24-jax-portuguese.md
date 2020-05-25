# Uma introdução a JAX
> Numpy + autograd + XLA
- toc: true 

JAX é uma nova biblioteca para Python da Google com foco em pesquisa de alta performance em Aprendizado de Máquina e seguindo o paradigma de programação funcional 
Mais especificamente JAX nos dá acesso a uma API compatível com numpy e scipy e transformações de função, as principais sendo grad, jit, vmap e pmap.

## O Wrapper de Numpy: jax.numpy
JAX nos dá acesso ao jax.numpy, uma reinplementação das funções do Numpy que são transformáveis pelas pelas trasformações de função do JAX.


```python
import jax.numpy as jnp
import numpy as np

a = np.array([1., 2., 3.])
b = np.array([1., 1., -1.])
print(np.dot(a, b), jnp.dot(a, b))
```

    0.0 0.0



```python
(np.square(a), jnp.square(a))
```




    (array([1., 4., 9.], dtype=float32), DeviceArray([1., 4., 9.], dtype=float32))



Note que JAX tem seu próprio tipo de array, o DeviceArray, em geral as funções vão castar arrays de numpy para DeviceArrays, então se você quiser boa performance é melhor fazer esse casting manualmente antes de passar os dados para várias funções.
Uma outra diferença é números aleatórios funcionam


```python

```

## Diferenciação Automática: jax.grad

Em aprendizado de máquina, principalmente quando estamos tratando de redes neurais, lidamos com muitas derivas, gradientes e afins: Para treinar uma regressão linear ou logística, precisamos computar um hessiano, para treinar uma rede neural usamos descida de gradiente, que requer o cálculo de um gradiente, dentre outros exemplos. 
Computar essas derivadas na mão é muitas vezes impossível (por questão de tempo), assim temos algoritmos como o backpropagation para redes neurais, porém se sempre tivessemos que implementar nós mesmos esse algoritmo, e implementar a derivada de cada uma das funções que vamos usar, terminaríamos com uma quatidade imensa de código duplicado, além duma imensa chance de errarmos algo na implementação e terminarmos sem conseguir bons resultados ou com resultados que não correspodem a realidade. 
Para lidar com isso temos diferenciação automática, simplesmente ter diferenciação automática para as funções de Numpy já é o bastante para uma biblioteca mostrar seu valor, e no caso existe uma biblioteca que é exatamente isso, chamada de Autograd, em muitos sentidos JAX é um sucessor dessa biblioteca, inclusive ambas têm muitos desenvolvedores em comum.


```python
from jax import grad
from math import pi, sqrt
dup = grad(jnp.square)
print(dup(3.0)) #A derivada de x² é 2x

def composite_func(x):
    y = x**2
    return jnp.cos(y)

g = grad(composite_func) # Pela regra da cadeia, dcos(x²)/dx = -2xsen(x²)
print(g(jnp.sqrt(pi/2)), -2*sqrt(pi/2))
```

    6.0
    -2.5066283 -2.5066282746310002


Para funções com várias variáveis de entrada a grad por padrão nos dá a derivada em função do primeiro parâmetro, mas podemos mudar isso com o argumento argnums. Também vale ressaltar que os argumentos não precisam ser apenas números e pode ser vetores


```python
def f(x, y):
    return x*(y**2)
dfdy = grad(f, argnums=(1))
print(dfdy(3.0, 4.0))
gradient = grad(f, argnums=(0, 1))
print(gradient(3.0, 4.0))

def g(v):
    return jnp.linalg.norm(v)
print(grad(g)(a))
```

    24.0
    (DeviceArray(16., dtype=float32), DeviceArray(24., dtype=float32))
    [0.26726124 0.5345225  0.8017837 ]


## Compilação com XLA: jit


```python

```


```python

```


```python

```

## Vetorização Automática: vmap


```python

```

## Paralelização: pmap


```python

```

## Sub bibliotecas experimentais


```python

```


```python

```

## O Ecosistema JAX
