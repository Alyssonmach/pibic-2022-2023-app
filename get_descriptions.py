def dcgan_description():
    
    description = 'A arquitetura **DCGAN (Deep Convolutional Generative Adversarial Network)** é uma '\
                  'técnica de aprendizado de máquina profunda para gerar imagens. Ela consiste em dois '\
                  'componentes principais: um gerador e um discriminador. O gerador tenta criar imagens '\
                  'que pareçam reais,enquanto o discriminador tenta identificar se as imagens geradas são '\
                  'reais ou falsas. Esses dois componentes trabalham juntos em um processo de treinamento '\
                  'iterativo, onde o gerador aprende a criar imagens cada vez mais realistas, e o '\
                  'discriminador aprende a identificar essas imagens cada vez melhor. A arquitetura DCGAN '\
                  'é baseada em Redes Neurais Convolucionais, que são tipicamente usadas para processamento '\
                  'de imagens.'
    
    return description

def dcgan_citation():

    citation = '* Radford, A., Metz, L., & Chintala, S.. (2015). Unsupervised Representation Learning '\
               'with Deep Convolutional Generative Adversarial Networks.'
    
    return citation

def mnist_description():

    description = '**MNIST** é uma base de dados de dígitos manuscritos amplamente utilizada em aprendizado '\
                  'de máquina e visão computacional. A base de dados contém 60.000 imagens de treinamento '\
                  'e 10.000 imagens de teste, cada uma com uma resolução de 28x28 pixels, contendo dígitos '\
                  'de 0 a 9 escritos à mão. As imagens são brancas e pretas, onde os pixels brancos '\
                  'representam o fundo e os pretos representam o traço do dígito. A base de dados é conhecida '\
                  'por ser simples, porém desafiadora, e é frequentemente utilizada como benchmark para '\
                  'algoritmos de reconhecimento de dígitos escritos à mão. A base de dados foi criada pelo '\
                  'Yann LeCun, Corinna Cortes e Christopher J.C. Burges, que publicou o trabalho em 1998.'
    
    return description

def mnist_citation():

    citation = '* Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. (1998). Gradient-based learning applied '\
               'to document recognition. Proceedings of the IEEE, 86(11), 2278-2324. doi: 10.1109/5.726791.'
    
    return citation

def fashion_mnist_description():

    description = '**Fashion-MNIST** é uma base de dados de imagens que contém 70.000 imagens em escala de cinza '\
                  'de 28x28 pixels de roupas e acessórios. Essa base de dados foi criada como uma alternativa '\
                  'à popular base de dados MNIST, que tem sido frequentemente usada como uma referência para testar '\
                  'algoritmos de visão computacional. A base de dados Fashion-MNIST é mais desafiadora do que a MNIST, '\
                  'pois as imagens de roupas e acessórios contêm mais variações de forma e textura do que os dígitos '\
                  'manuscritos na base de dados MNIST. Além disso, a Fashion-MNIST oferece uma oportunidade para avaliar '\
                  'o desempenho de algoritmos de visão computacional em aplicações relacionadas à moda, como reconhecimento '\
                  'de vestuário, recomendação de estilo e muito mais. Desde sua criação, a Fashion-MNIST se tornou uma base de '\
                  'dados amplamente utilizada na comunidade de aprendizado de máquina, e muitos algoritmos de classificação e '\
                  'redes neurais profundas foram testados em seus dados.'
    
    return description

def fashion_mnist_citation():

    citation = '* Han Xiao, Kashif Rasul, Roland Vollgraf. (2017). Fashion-MNIST: a Novel Image Dataset for '\
               'Benchmarking Machine Learning Algorithms. arXiv preprint arXiv:1708.07747.'
    

    return citation

def emnist_letters_description():

    description = '**EMNIST** é uma base de dados de imagens de dígitos e caracteres alfanuméricos manuscritos '\
                  'em inglês, que contém mais de 800.000 imagens em escala de cinza de 28x28 pixels. Ela foi '\
                  'criada como uma expansão da popular base de dados MNIST, adicionando letras maiúsculas e '\
                  'minúsculas, bem como números e símbolos não numéricos. Essa base de dados foi criada para '\
                  'fornecer uma tarefa de classificação mais desafiadora para algoritmos de reconhecimento de '\
                  'caracteres, comparado com a MNIST, permitindo uma avaliação mais abrangente de algoritmos de '\
                  'aprendizado de máquina em aplicações de OCR (Optical Character Recognition) e reconhecimento '\
                  'de escrita manual. Além disso, a EMNIST também é útil para a análise de assinaturas manuscritas, '\
                  'reconhecimento de placas de veículos e outras aplicações de processamento de texto. A EMNIST é '\
                  'amplamente utilizada na comunidade de aprendizado de máquina e tem sido um ponto de referência '\
                  'para muitos estudos e pesquisas.'

    return description

def emnist_letters_citation():

    citation = '* Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten '\
               'letters. arXiv preprint arXiv:1702.05373.'

    return citation

def cifar10_description():

    description = '**CIFAR-10** é uma base de dados de imagens criada para fins de reconhecimento de objetos. Ela contém 60.000 '\
                  'imagens coloridas de 32x32 pixels, divididas em 10 classes, com 6.000 imagens por classe. As 10 classes '\
                  'incluem avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio e caminhão. A base de dados '\
                  'é frequentemente utilizada como uma tarefa de classificação de imagem de referência para algoritmos de '\
                  'aprendizado de máquina e visão computacional. Por causa de sua pequena resolução, a CIFAR-10 é '\
                  'frequentemente considerada uma tarefa mais fácil do que outras bases de dados de reconhecimento de '\
                  'objetos, como a ImageNet, mas ainda desafiadora o suficiente para permitir que os pesquisadores '\
                  'desenvolvam e avaliem algoritmos de reconhecimento de objetos mais avançados. A CIFAR-10 é '\
                  'amplamente utilizada na comunidade de aprendizado de máquina e visão computacional, e tem '\
                  'sido um ponto de referência para muitos estudos e pesquisas.'
    
    return description

def cifar10_citation():
     
     citation = '* Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images. '\
                'Technical report.'

     return citation 
    