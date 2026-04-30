# Explique por que o número de épocas de treinamento varia a cada vez que executamos o treinamento do perceptron.

- O número de épocas varia porque os pesos iniciais começam com valores aleatórios toda vez. Como o ponto de partida muda a cada execução, o perceptron pode levar mais ou menos passos (épocas) para conseguir ajustar a reta e separar os dados corretamente.

# Qual a principal limitação do perceptron quando aplicado em problemas de classificação de padrões.

-  principal limitação do perceptron é que ele só resolve problemas que são linearmente separáveis. Na prática, isso significa que ele só consegue separar os dados usando uma linha reta. Se o problema for mais complexo e precisar de uma curva para separar as classes (como no caso do XOR), ele simplesmente trava e nunca chega na solução correta.
