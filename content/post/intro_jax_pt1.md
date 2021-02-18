+++
author = "João G. M. Araújo"
date = "2020-05-25"
hasMath = false
title = "Uma introdução a JAX"
subtitle = "Numpy + autograd + XLA"
+++ 

[JAX](https://github.com/google/jax) é uma nova biblioteca para Python da Google com foco em pesquisa de alta performance em Aprendizado de Máquina e seguindo o paradigma de programação funcional. 
Mais especificamente JAX nos dá acesso a uma API compatível com numpy e scipy e transformações de função, as principais sendo grad, jit, vmap e pmap(que vai ter seu próprio post no futuro).

## O Wrapper de Numpy: jax.numpy
JAX nos dá acesso ao jax.numpy, uma reimplementação das funções do Numpy.


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




    (array([1., 4., 9.]), DeviceArray([1., 4., 9.], dtype=float32))



Note que JAX tem seu próprio tipo de array, o DeviceArray, em geral as funções vão castar arrays de numpy para DeviceArrays, então se você quiser boa performance é melhor fazer esse casting manualmente antes de passar os dados para várias funções.
Uma outra diferença é como números aleatórios funcionam. JAX não tem a jax.numpy.random, em vez disso ele tem a sua própria sub-biblioteca jax.random

## Números aleatórios jax.random

Uma das partes mais peculiares de JAX, para faciliar implementações usando paralelismo não existe uma semente global para geradores de números aleatórios, em vez disso em JAX você passa explicitamente a seed para cada função que envolve aleatoriedade, e cabe a você atualizá-la


```python
import jax
key = jax.random.PRNGKey(42) #cria um semente aleatória
a = jax.random.normal(key, ())
b = jax.random.normal(key, ())
print(a, b)
print(a == b) #como usamos a mesma semente para a mesma função temos valores iguais
k1, k2 = jax.random.split(key, 2) #vamos criar duas novas seeds a partir da primeira
a = jax.random.normal(k1, ())
b = jax.random.normal(k2, ())
print(a, b) #Agora são diferentes
```

    -0.18471177 -0.18471177
    True
    0.13790321 1.3694694


## Transformações

O principal diferencial de JAX são suas tranformações de funções, que nos permitem modificar facilmente funções definidas a partir de outras funções do JAX e algumas primitivas de Python. Algo muito útil e legal delas é que elas podem ser utilizadas em conjunto, nos permitindo por exemplo compilar a derivada de uma função vetorizada apenas aplicando 3 transformações uma seguida da outra a função original. 
Porém existem alguns cuidados que dever ser tomados ao se usar esses transformações, para entender esse cuidados melhor cheque esse [link](https://github.com/google/jax#current-gotchas) e abra o notebook

### Diferenciação Automática: jax.grad

Em aprendizado de máquina, principalmente quando estamos tratando de redes neurais, lidamos com muitas derivadas, gradientes e afins: Para treinar uma regressão linear ou logística, precisamos computar um hessiano, para treinar uma rede neural usamos descida de gradiente, que requer o cálculo de um gradiente, dentre outros exemplos. 
Computar essas derivadas na mão é muitas vezes impossível (por questão de tempo), assim temos algoritmos como o backpropagation para redes neurais, porém se sempre tivessemos que implementar nós mesmos esse algoritmo, e implementar a derivada de cada uma das funções que vamos usar, terminaríamos com uma quatidade imensa de código duplicado, além duma imensa chance de errarmos algo na implementação e terminarmos sem conseguir bons resultados ou com resultados que não correspodem a realidade. 
Para lidar com isso temos diferenciação automática, simplesmente ter diferenciação automática para as funções de Numpy já é o bastante para uma biblioteca mostrar seu valor, e no caso existe uma biblioteca que é exatamente isso, chamada de Autograd, em muitos sentidos JAX é um sucessor dessa biblioteca, inclusive ambas têm muitos desenvolvedores em comum.


```python
from jax import grad
from math import pi, sqrt
dup = grad(jnp.square)
print(dup(3.0)) #A derivada de x² é 2x
print(grad(dup)(3.0)) #Podemos aplicar várias vezes a grad

def composite_func(x):
    y = x**2
    return jnp.cos(y)

g = grad(composite_func) # Pela regra da cadeia, dcos(x²)/dx = -2xsen(x²)
print(g(jnp.sqrt(pi/2)), -2*sqrt(pi/2))
```

    6.0
    2.0
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
    1.0


### Compilação com XLA: jit

Mas as vantagens de jax não param em diferenciação automática, se não seria apenas um clone do autograd, jax também tem a habilidade de compilar funções usando o XLA (accelerated linear algebra) da Google, tornando-as bem mais rápidas, além de permitir o uso de aceleradores como GPUs e TPUs.


```python
dot = jax.jit(jnp.dot)
```


```python
a = jax.random.normal(k1, (2024, 2024))
b = jax.random.normal(k2, (2024, 2024))
```


```python
%%timeit
np.dot(a, b)
```

    1 loop, best of 3: 256 ms per loop



```python
%%timeit
dot(a, b)
```

    The slowest run took 1255.15 times longer than the fastest. This could mean that an intermediate result is being cached.
    1 loop, best of 3: 169 µs per loop


## Vetorização Automática: vmap

Vmap é uma transformação muito interessante, usando ela é possível vetorizar automaticamente nossas funções, ou seja, em vez de ter que fazer uma função que lida com um batch de dados, podemos fazer uma função que recebe um único dado e depois usar a trasformação para ganhar a versão que lida com o batch.


```python
a = np.array([1., 2., 3.])
b = np.array([1., 1., -1.])
c = np.array([[1., 2., 3.], [4., 5., 6.]])

@jax.vmap #Podemos usar as transformações como decoradores
def f(x, y):
    return x/y + 1.
print(f(a, b))

def prod(x, y):
    return x@y
print(prod(a, b))
```

    [ 2.  3. -2.]
    0.0



```python
prod(a, c) #a e c não têm dimensões compatíveis
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-11-908e821a683b> in <module>()
    ----> 1 prod(a, c) #a e c não têm dimensões compatíveis
    

    <ipython-input-10-7f441aca0238> in prod(x, y)
          9 
         10 def prod(x, y):
    ---> 11     return x@y
         12 print(prod(a, b))


    ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 2 is different from 3)



```python
batch_prod = jax.vmap(prod, in_axes=(None, 0)) #vamos multiplica a por cada linha de c
batch_prod(a, c)
```




    DeviceArray([14., 32.], dtype=float32)



Nessa primeira parte vimos qual o propósito da biblioteca e suas principais funções, nos próximos posts vamos explorar como criar redes neurais com jax, suas bibliotecas experimentais, o ecossistema de bibliotecas escritas usando jax e a pmap
