import streamlit as st
import torch
import utils

def site_configurations():
    '''
    Configures the appearance of a Streamlit app by setting the page title, layout, hiding certain elements,
    and setting padding for the main container.
    '''

    st.set_page_config(page_title = 'PIBIC GANs', layout = 'wide')
    st.markdown("<div id = 'inicio' style = 'visibility: hidden'></div>", unsafe_allow_html = True)
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

def site_header():
    '''
    Creates a Streamlit header and markdown content describing the objective of a research project on generating
    artificial images using Generative Adversarial Networks (GANs) and other generative algorithms, and the purpose
    of the site for displaying the results of the project.
    '''

    st.header(body = 'Geração de Imagens Artificiais para Aumento de Dados Utilizando Redes Adversárias '
                 'Generativas (GANs)')
    st.markdown(body = '**Objetivo da Pesquisa**: O objetivo desse projeto é o desenvolvimento de um sistema '
                    'que permita o aumento de dados em bases de imagens a partir da geração de amostras '
                    'sintéticas utilizando GANs e outros algoritmos de Sistemas Generativos. Também é '
                    'esperado que o sistema gere imagens artificiais com uma boa representação visual, '
                    'diversidade e fidelidade com o grau de detalhamento presente nas imagens originais '
                    'das bases de dados testadas. Esse site tem o objetivo de ilustrar todos os resultados '
                    'obtidos com as redes generativas criadas. Essa pesquisa está vinculada a um trabalho '
                    'de iniciação científica PIBIC para alunos de graduação da Universidade Federal de Campina ' 
                    'Grande (UFCG).')

def members_project():
    '''
    Function that displays information about project members.

    The function uses the streamlit library expander to create a component that can be expanded by the user
    to display information about project members. Information includes name, education and a brief
    biography. To display the information, the function uses the streamlit library to create a column for the image
    of the member and another column for information about the member.
    '''

    with st.expander(label = '**Clique aqui para saber mais sobre os membros do projeto**'):

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

def select_model_and_data():
    '''
    This function creates two dropdowns to select a generative model and a dataset to be used for image generation.
    If "Simple GAN" is selected as the generative model, the function only displays datasets that are compatible with 
    this specific model. The function returns the selected generative model and dataset.

    Returns:
    algorithm_box (str): The selected generative model.
    data_box (str): The selected dataset to be used for image generation.
    '''
    
    col1, col2 = st.columns(spec = [1, 1])

    with col1:

        algorithms = ['Simple GAN (Indisponível)', 'DCGAN', 'SNGAN (Indisponível)', 'WGAN - GP (Indisponível)',
                    'SNGAN + WGAN - GP (Indisponível)', 'Variational Autoencoder (Indisponível)']
        algorithm_box = st.selectbox(label = '**Selecione o Sistema Generativo**:', 
                                    options = algorithms, index = 1, 
                                    help = 'Selecione o **Sistema Generativo** para gerar as imagens artificiais '
                                        'com base no **Conjunto de Imagens** selecionado.')

    with col2:

        datasets_simple_gan = ['MNIST', 'Fashion MNIST', 'EMNIST Letters']
        datasets = ['MNIST', 'Fashion MNIST', 'EMNIST Letters', 'CIFAR10 (Indisponível)', 'CELEBA (Indisponível)']

        if algorithm_box == 'Simple GAN (Indisponível)':
            data_box = st.selectbox(label = '**Selecione o Conjunto de Imagens**:', options = datasets_simple_gan, index = 0, 
                                    help = 'Selecione o **Conjunto de Imagens** para testar a capacidade de geração de imagens '
                                        'do **Sistema Generativo** escolhido.')
        else:
            data_box = st.selectbox(label = '**Selecione o Conjunto de Imagens**:', options = datasets, index = 0, 
                                    help = 'Selecione o **Conjunto de Imagens** para testar a capacidade de geração de imagens '
                                        'do **Sistema Generativo** escolhido.')
    
    col1, col2 = st.columns(spec = [1, 1])
    
    with col1:
        st.markdown(body = f'### Sistema Generativo: ```{algorithm_box}```')
    with col2:
        st.markdown(body = f'### Conjunto de Imagens: ```{data_box}```')
    st.markdown(body = '***')

    return algorithm_box, data_box

def area_with_interpolation_images(start_plot_number, end_plot_number, z_dim, n_classes, device, gen,
                                   n_interpolation, column1, column2):
    '''
    This function generates interpolated images between two class labels using the generator model 
    passed as input. The user can select the start and end classes, the size of the noise vector, 
    the number of classes, the device used for training, and the number of interpolation images to 
    be generated. The images are displayed in a Streamlit app in two columns. Each column displays 
    two sets of interpolated images. The function does not return any values.

    Parameters:
        start_plot_number (int): First class label for image interpolation.
        end_plot_number (int): Second class label for image interpolation.
        z_dim (int): Size of the input noise vector.
        n_classes (int): Number of classes.
        device (str): Device used for training (cpu or gpu).
        gen (nn.Module): Generator model.
        n_interpolation (int): Number of interpolation images to be generated.
        column1 (Streamlit column object): First column object where the images will be displayed.
        column2 (Streamlit column object): Second column object where the images will be displayed.
    '''

    with column1:
        images = utils.interpolate_class(first_number = start_plot_number, 
                                         second_number = end_plot_number, z_dim = z_dim, 
                                         n_classes = n_classes, device = device, gen = gen, 
                                         n_interpolation = n_interpolation, show_tensor = False)
        utils.create_gif(images_pytorch_tensor = images, image_location = 'images/tensor_images1.gif')
        
        st.image(image = 'images/tensor_images1.gif', width = 150)

        images = utils.interpolate_class(first_number = start_plot_number, 
                                         second_number = end_plot_number, z_dim = z_dim, 
                                         n_classes = n_classes, device = device, gen = gen, 
                                         n_interpolation = n_interpolation, show_tensor = False)
        utils.create_gif(images_pytorch_tensor = images, image_location = 'images/tensor_images2.gif')
        
        st.image(image = 'images/tensor_images2.gif', width = 150)
    
    with column2:
        images = utils.interpolate_class(first_number = start_plot_number, 
                                         second_number = end_plot_number, z_dim = z_dim, 
                                         n_classes = n_classes, device = device, gen = gen, 
                                         n_interpolation = n_interpolation, show_tensor = False)
        utils.create_gif(images_pytorch_tensor = images, image_location = 'images/tensor_images3.gif')

        st.image(image = 'images/tensor_images3.gif', width = 150)

        images = utils.interpolate_class(first_number = start_plot_number, 
                                         second_number = end_plot_number, z_dim = z_dim, 
                                         n_classes = n_classes, device = device, gen = gen, 
                                         n_interpolation = n_interpolation, show_tensor = False)
        utils.create_gif(images_pytorch_tensor = images, image_location = 'images/tensor_images4.gif')
        
        st.image(image = 'images/tensor_images4.gif', width = 150)

    st.markdown(body = '***')

def area_with_images_generated(n_noise, n_classes, n_interpolation, z_dim, gen, device, column1, column2):
    '''
    Displays interpolated images generated by the GAN model using random noise.

    Parameters:
        n_noise (int): The number of noise vectors to generate.
        n_classes (int): The number of classes used in the GAN model.
        n_interpolation (int): The number of images to generate between the start and end noise vectors.
        z_dim (int): The dimension of the input noise vector.
        gen (nn.Module): The generator module of the GAN model.
        device (str): Device used for training (cpu or gpu).
        column1 (Streamlit column object): The first column in the Streamlit app.
        column2 (Streamlit column object): The second column in the Streamlit app.
    '''
    
    interpolation_label = utils.get_one_hot_labels(torch.Tensor([n_noise]).long(), 
                                                   n_classes).repeat(n_interpolation, 1).float()

    with column1:
        utils.interpolate_noise_gif(n_noise = n_noise, z_dim = z_dim, n_interpolation = n_interpolation, 
                                    interpolation_label = interpolation_label, gen = gen, device = device)
        st.image(image = 'images/interpolate-noise.gif', width = 300) 

    with column2:
        utils.interpolate_noise_gif(n_noise = n_noise, z_dim = z_dim, n_interpolation = n_interpolation, 
                                    interpolation_label = interpolation_label, gen = gen, device = device)
        st.image(image = 'images/interpolate-noise.gif', width = 300) 