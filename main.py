import get_descriptions
import streamlit as st
import torch
import utils

st.set_page_config(page_title = 'PIBIC GANs', layout = 'wide', page_icon = '🎨')
st.markdown("<div id='inicio' style='visibility: hidden'></div>", unsafe_allow_html = True)
st.markdown('<style> #MainMenu {visibility: hidden;} footer {visibility: hidden;} </style>', 
            unsafe_allow_html = True)

padding = 1
st.markdown(f"""<style>
                .reportview-container .main .block-container{{
                    padding-top: {padding}rem;
                    padding-right: {padding}rem;
                    padding-left: {padding}rem;
                    padding-bottom: {padding}rem;}}
                </style>""", unsafe_allow_html = True)

st.header(body = 'Geração de Imagens Artificiais para Aumento de Dados Utilizando Redes Adversárias '
                 'Generativas (GANs)')
st.markdown(body = '**Objetivo da Pesquisa**: O objetivo desse projeto é o desenvolvimento de um sistema '
                   'que permita o aumento de dados em bases de imagens a partir da geração de amostras '
                   'sintéticas utilizando GANs e outros algoritmos de Sistemas Generativos. Também é '
                   'esperado que o sistema gere imagens artificiais com uma boa representação visual, '
                   'diversidade e fidelidade com o grau de detalhamento presente nas imagens originais '
                   'das bases de dados testadas.')

with st.expander(label = 'Clique aqui para saber mais sobre os membros do projeto'):
        st.markdown(body = '***')

        col1, col2 = st.columns(spec = [1,8])

        with col1:
            st.image(image = 'https://user-images.githubusercontent.com/58775072/145095066-08ec2fd0-0e30-44b4-998d-c3e78d4409e0.jpg', 
                     width = 120)
        with col2:
            st.markdown(body = '**Nome**: [Luciana Ribeiro Veloso](http://lattes.cnpq.br/2498050002491677)')
            st.markdown(body = '**Formação Acadêmica**: Doutora em Engenharia Elétrica (UFCG)')
            st.markdown(body = '**Biografia**: Possui graduação em Engenharia Elétrica pela Universidade Federal da '
                               'Paraíba (1995), mestrado em Engenharia Elétrica pela Universidade Federal da Paraíba '
                               '(1998) e Doutorado em Engenharia Elétrica pela Universidade Federal de Campina Grande '
                               '(2009). Atualmente é professora da Universidade Federal de Campina Grande. Tem '
                               'experiência na área de Engenharia Elétrica, com ênfase em Processamento de Imagens, ' 
                               'atuando principalmente nos seguintes temas: Reconhecimento de Palavras Manuscritas, '
                               'Processamento de Imagens e Reconhecimento de Padrões.')

        st.markdown(body = '***')
        
        col1, col2 = st.columns(spec = [1,8])
        
        with col1:
            st.image(image = 'https://user-images.githubusercontent.com/58775072/145095291-61c00da5-211a-4b86-9728-77594fbf60e9.gif', 
                     width = 120)
        with col2:
            st.markdown(body = '**Nome**: [Leo de Lima Araújo](http://lattes.cnpq.br/2093156188518982)')
            st.markdown(body = '**Formação Acadêmica**: Mestrando em Engenharia Elétrica (UFCG)')
            st.markdown(body = '**Biografia**: Pós-graduando em Engenharia Elétrica pela Universidade Federal de Campina '
                               'Grande. Trabalha sobretudo com Aprendizado Profundo (*Deep Learning*) e Visão Computacional '
                               'e atua no desenvolvimento pesquisas na área de saúde, envolvendo o diagnóstico automatizado '
                               'de Distúrbios Pulmonares e *COVID-19*, além de aplicações industriais, identificando defeitos '
                               'em televisores/monitores na linha de produção.')
        
        st.markdown(body = '***')
        
        col1, col2 = st.columns(spec = [1,8])
        
        with col1:
            st.image(image = 'https://user-images.githubusercontent.com/58775072/145095535-fb303146-1c30-40bf-85d2-6b90e4a68ab3.png', 
                     width = 120)
        with col2:
            st.markdown(body = '**Nome**: [Alysson Machado de Oliveira Barbosa](http://lattes.cnpq.br/0536933202710340)')
            st.markdown(body = '**Formação Acadêmica**: Graduando em Engenharia Elétrica (UFCG)')
            st.markdown(body = '**Biografia**: Possui segundo grau completo na instituição Colégio Alfredo Dantas (CAD) '
                               'e atualmente faz graduação em Engenharia Elétrica pela Universidade Federal de Campina '
                               'Grande (UFCG). Desenvolveu um projeto intitulado "Álgebra Linear com Python" , visando aplicar '
                               'computacionalmente conceitos de Álgebra Linear. Atualmente é ativo no desenvolvimento '
                               'de trabalhos em Inteligência Artificial (IA) com conhecimentos em *Machine Learning* ' 
                               'e *Deep Learning*. Tem experiência no desenvolvimento de páginas web, programação em C, '
                               'C++, Python e versionamento de projetos com Git. Desenvolveu atividades relevantes '
                               'para a comunidade acadêmica como o evento *Academic Talk* em parceria com o capítulo '
                               'PELS/IAS da UFCG associado ao Instituto de Engenheiros Elétricos e Eletrônicos (IEEE).')

        st.markdown(body = '***')

data_box = st.selectbox(label = 'Selecione a Base de Dados:', options = ['MNIST', 'CELEBA (Indisponível)', 
                        'CIFAR - 10 (Indisponível)', 'FASHION - MNIST (Indisponível)'], index = 0, 
                        help = 'Selecione a base de dados para testar a capacidade dos algoritmos '
                        'generativos disponíveis para teste.')
algorithm_box = st.selectbox(label = 'Selecione o Sistema Generativo:', 
                             options = ['Simple GAN (Indisponível)', 'DCGAN', 'SNGAN (Indisponível)', 
                             'WGAN - GP (Indisponível)', 'SNGAN + WGAN - GP (Indisponível)'], index = 1, 
                             help = 'Selecione o sistema generativo para visualizar as imagens '
                             'artificiais geradas de acordo com a base de dados selecionada.')

st.markdown(body = f'{data_box} e {algorithm_box}')

st.markdown(body = f'### Base de Dados: {data_box} | Sistema Generativo: {algorithm_box}')
st.markdown(body = '***')

if data_box == 'MNIST' and algorithm_box == 'DCGAN':

    z_dim = 64
    mnist_shape = (1, 28, 28)
    n_classes = 10
    device = 'cpu'

    generator_input_dim, discriminator_im_chan = utils.get_input_dimensions(z_dim, mnist_shape, n_classes)
    gen = utils.Generator(input_dim = generator_input_dim).to(device)
    gen.load_state_dict(torch.load('models/gen_conditional_dcgan.pth', 
                        map_location = torch.device(device)))
    gen.eval()

    col1, col2, col3 = st.columns(spec = [5, 1, 1])

    with col1:
        start_plot_number = st.slider(label = 'Selecione o primeiro dígito para interpolação:', 
                                      min_value = 0, max_value = 9, value = 3)
        end_plot_number = st.slider(label = 'Selecione o segundo dígito para interpolação:',
                                    min_value = 0, max_value = 9, value = 9)
        n_interpolation = st.slider(label = 'Selecione a quantidade de imagens geradas na interpolação:',
                                    min_value = 10, max_value = 200, value = 100)
    
    with col2:
        images = utils.interpolate_class(first_number = start_plot_number, 
                                         second_number = end_plot_number, z_dim = z_dim, 
                                         n_classes = n_classes, device = device, gen = gen, 
                                         n_interpolation = n_interpolation, show_tensor = False)
        utils.create_gif(images_pytorch_tensor = images)
        
        st.image(image = 'images/tensor_images.gif', width = 150)

        images = utils.interpolate_class(first_number = start_plot_number, 
                                         second_number = end_plot_number, z_dim = z_dim, 
                                         n_classes = n_classes, device = device, gen = gen, 
                                         n_interpolation = n_interpolation, show_tensor = False)
        utils.create_gif(images_pytorch_tensor = images)
        
        st.image(image = 'images/tensor_images.gif', width = 150)
    
    with col3:
        images = utils.interpolate_class(first_number = start_plot_number, 
                                         second_number = end_plot_number, z_dim = z_dim, 
                                         n_classes = n_classes, device = device, gen = gen, 
                                         n_interpolation = n_interpolation, show_tensor = False)
        utils.create_gif(images_pytorch_tensor = images)

        st.image(image = 'images/tensor_images.gif', width = 150)

        images = utils.interpolate_class(first_number = start_plot_number, 
                                         second_number = end_plot_number, z_dim = z_dim, 
                                         n_classes = n_classes, device = device, gen = gen, 
                                         n_interpolation = n_interpolation, show_tensor = False)
        utils.create_gif(images_pytorch_tensor = images)
        
        st.image(image = 'images/tensor_images.gif', width = 150)
    
    st.markdown(body = '***')

    col1, col2, col3 = st.columns(spec = [2, 1, 1])

    with col1:
        n_interpolation = st.slider(label = 'Dígite a quantidade de imagens geradas na interpolação:', 
                                    min_value = 6, max_value = 30, value = 9, step = 3)
        n_noise = st.slider(label = 'Selecione o dígito a ser gerado:', min_value = 0, max_value = 9,
                            value = 9)
        interpolation_label = utils.get_one_hot_labels(torch.Tensor([n_noise]).long(), 
                                                       n_classes).repeat(n_interpolation, 1).float()

    with col2:
        utils.interpolate_noise_gif(n_noise = n_noise, z_dim = z_dim, n_interpolation = n_interpolation, 
                                    interpolation_label = interpolation_label, gen = gen, device = device)
        st.image(image = 'images/interpolate-noise.gif', width = 300) 

    with col3:
        utils.interpolate_noise_gif(n_noise = n_noise, z_dim = z_dim, n_interpolation = n_interpolation, 
                                    interpolation_label = interpolation_label, gen = gen, device = device)
        st.image(image = 'images/interpolate-noise.gif', width = 300) 
    
    st.markdown(body = '#### Detalhes sobre o Sistema Generativo:')
    st.markdown(body = get_descriptions.dcgan_description())
    st.image(image = 'data/dcgan-gen.png', caption = 'Desenho arquitetural do gerador da DCGAN '
             'Radford et al (2016). Disponível em: https://arxiv.org/pdf/1511.06434v1.pdf.')
    st.markdown(body = '***')
    st.markdown(body = '#### Detalhes sobre a Base de Dados:')
    st.markdown(body = get_descriptions.mnist_description())
else:
    st.warning(body = 'Desculpe, mas essa combinação de base de dados e sistema generativo está indisponível '
               'no momento.')
