def simple_gan_description():
    '''
    Get Simple GAN architecture description.
    '''

    description = 'Uma **Rede Adversária Generativa Simples (Simple GAN)** é uma arquitetura de rede neural que consiste em '\
                  'duas redes: um gerador e um discriminador. O gerador gera amostras de dados sintéticos que imitam os dados reais, '\
                  'enquanto o discriminador avalia a autenticidade das amostras. Durante o treinamento, as duas redes são '\
                  'treinadas simultaneamente em um processo competitivo, no qual o gerador tenta enganar o discriminador, '\
                  'enquanto o discriminador tenta distinguir as amostras geradas do conjunto de dados reais. Com o tempo, o '\
                  'gerador aprende a produzir amostras mais realistas, enquanto o discriminador se torna mais preciso em distinguir '\
                  'entre amostras falsas e reais.'
    
    return description 

def simple_gan_citation():
    '''
    Get Simple GAN architecture citation in APA format.
    '''

    citation = '* Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). '\
               'Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).'
    return citation 

def dcgan_description():
    '''
    Get DCGAN architecture description.
    '''
    
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
    '''
    Get DCGAN architecture citation in APA format.
    '''

    citation = '* Radford, A., Metz, L., & Chintala, S.. (2015). Unsupervised Representation Learning '\
               'with Deep Convolutional Generative Adversarial Networks.'
    
    return citation

def sngan_description():
    '''
    Get SNGAN architecture description.
    '''

    description = 'A arquitetura **SNGAN** é uma variação da arquitetura GAN que utiliza normalização espectral '\
                  'na rede discriminadora e Path Length Regularization na rede geradora para gerar imagens '\
                  'mais nítidas e realistas. A normalização espectral ajuda a estabilizar o treinamento da '\
                  'rede e melhorar a qualidade das imagens, enquanto o mecanismo de auto-correção da rede '\
                  'geradora garante a consistência das imagens geradas. A abordagem SNGAN tem sido amplamente '\
                  'utilizada em aplicações de geração de imagens, como em jogos de vídeo e animação digital.'

    return description

def sngan_citation():
    '''
    Get SNGAN architecture citation in APA format.
    '''

    citation = '* Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). Spectral Normalization for Generative '\
               'Adversarial Networks. CoRR, abs/1802.05957. http://arxiv.org/abs/1802.05957.'

    return citation

def wgan_gp_description():
    '''
    Get WGAN-GP architecture description.
    '''

    description = 'A arquitetura **WGAN-GP (Wasserstein Generative Adversarial Network with Gradient Penalty)** é uma extensão '\
                  'da arquitetura GAN que utiliza a distância de Wasserstein para medir a diferença entre as distribuições de '\
                  'probabilidade da rede geradora e a rede discriminadora. Essa abordagem permite uma convergência mais estável '\
                  'e confiável durante o treinamento da rede, resultando em imagens geradas de maior qualidade. Além disso, a '\
                  'WGAN-GP também utiliza uma penalidade de gradiente para a rede discriminadora, o que ajuda a evitar problemas '\
                  'de instabilidade no treinamento e melhora a qualidade das imagens geradas. A combinação dessas técnicas torna '\
                  'a WGAN-GP uma arquitetura muito popular para geração de imagens de alta qualidade em aplicações como jogos de '\
                  'vídeo e animação digital.'

    return description

def wgan_gp_citation():
    '''
    Get WGAN-GP architecture citation in APA format.
    '''

    citation = '* Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. C. (2017). Improved Training of Wasserstein '\
               'GANs. In Advances in Neural Information Processing Systems (pp. 5769-5779).'

    return citation

def wgan_gp_sngan_description():
    '''
    Get WGAN-GP + SNGAN architecture description.
    '''

    description = 'Uma união das ideias da arquitetura **SNGAN** e **WGAN-GP** pode resultar em uma nova arquitetura de GAN que incorpora '\
                  'a normalização espectral e a regularização de caminho de comprimento da SNGAN com a distância de Wasserstein e a '\
                  'penalidade de gradiente da WGAN-GP. Essa combinação pode melhorar ainda mais a qualidade das imagens geradas pela '\
                  'rede, garantindo uma convergência mais estável.'

    return description

def mnist_description():
    '''
    Get MNIST dataset description.
    '''

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
    '''
    Get MNIST dataset citation in APA format.
    '''

    citation = '* Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. (1998). Gradient-based learning applied '\
               'to document recognition. Proceedings of the IEEE, 86(11), 2278-2324. doi: 10.1109/5.726791.'
    
    return citation

def fashion_mnist_description():
    '''
    Get Fashion MNIST dataset description.
    '''

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
    '''
    Get Fashion MNIST dataset citation in APA format.
    '''

    citation = '* Han Xiao, Kashif Rasul, Roland Vollgraf. (2017). Fashion-MNIST: a Novel Image Dataset for '\
               'Benchmarking Machine Learning Algorithms. arXiv preprint arXiv:1708.07747.'
    

    return citation

def emnist_letters_description():
    '''
    Get EMNIST Letters dataset description.
    '''

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
    '''
    Get EMNIST Letters dataset citation in APA format.
    '''

    citation = '* Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten '\
               'letters. arXiv preprint arXiv:1702.05373.'

    return citation

def celeba_description():
    '''
    Get CelebA dataset description.
    '''

    description = '**CelebA** é um conjunto de mais de 200.000 imagens de rostos de celebridades, com anotações que '\
                  'incluem a posição das feições faciais, a presença de óculos e outras características. A base de dados'\
                  ' é amplamente utilizada para treinar e avaliar algoritmos de reconhecimento facial, classificação de '\
                  'gênero e análise de expressões faciais. Além disso, a CelebA é uma das bases de dados mais populares '\
                  'para a geração de imagens de faces por meio de técnicas de aprendizado profundo, sendo utilizada em '\
                  'diversos estudos de redes neurais generativas.'
    
    return description

def celeba_citation():
    '''
    Get CelebA dataset citation in APA format.
    '''

    citation = '* Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep learning face attributes in the wild. '\
               'In Proceedings of the IEEE International Conference on Computer Vision (pp. 3730-3738).'
    
    return citation 
    