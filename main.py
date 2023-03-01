import get_descriptions
import streamlit as st
import site_structure
import generators
import torch
import utils

site_structure.site_configurations()
site_structure.site_header()
site_structure.members_project()

algorithm_box, data_box = site_structure.select_model_and_data()
device = 'cpu'

if data_box == 'MNIST' and algorithm_box == 'DCGAN':

    z_dim, mnist_shape, n_classes = 64, (1, 28, 28), 10

    generator_input_dim = utils.get_input_dimensions(z_dim, n_classes)
    gen = generators.GeneratorDCGAN_V1_0(input_dim = generator_input_dim).to(device)
    gen.load_state_dict(torch.load('models/gen_conditional_dcgan_mnist.pth', 
                        map_location = torch.device(device)))
    gen.eval()

    col1, col2, col3 = st.columns(spec = [5, 1, 1])

    with col1:
        start_plot_number = st.slider(label = 'Selecione o primeiro dígito para interpolação:', 
                                      min_value = 0, max_value = 9, value = 3)
        st.markdown(body = f'**Primeiro Digito Selecionado**: ```{start_plot_number}```')
        end_plot_number = st.slider(label = 'Selecione o segundo dígito para interpolação:',
                                    min_value = 0, max_value = 9, value = 9)
        st.markdown(body = f'**Segundo Digito Selecionado**: ```{end_plot_number}```')
        n_interpolation = st.slider(label = 'Selecione a quantidade de imagens geradas na interpolação:',
                                    min_value = 10, max_value = 200, value = 100)
    
    site_structure.area_with_interpolation_images(start_plot_number = start_plot_number, 
                                                  end_plot_number = end_plot_number,
                                                  z_dim = z_dim, n_classes = n_classes, device = device, 
                                                  gen = gen, n_interpolation = n_interpolation, column1 = col2, 
                                                  column2 = col3)

    col1, col2, col3 = st.columns(spec = [2, 1, 1])

    with col1:
        n_interpolation = st.slider(label = 'Dígite a quantidade de imagens geradas na interpolação:', 
                                    min_value = 6, max_value = 30, value = 9, step = 3)
        n_noise = st.slider(label = 'Selecione o dígito a ser gerado:', min_value = 0, max_value = 9,
                            value = 9)
        st.markdown(body = f'**Digito Selecionado**: ```{n_noise}```')

    site_structure.area_with_images_generated(n_noise = n_noise, n_classes = n_classes, 
                                              n_interpolation = n_interpolation, z_dim = z_dim, 
                                              gen = gen, device = device, column1 = col2, column2 = col3)
    
    st.markdown(body = '#### Detalhes sobre o Sistema Generativo:')
    st.markdown(body = get_descriptions.dcgan_description())
    st.markdown(body = get_descriptions.dcgan_citation())
    st.markdown(body = '***')
    st.markdown(body = '#### Detalhes sobre a Base de Dados:')
    st.markdown(body = get_descriptions.mnist_description())
    st.markdown(body = get_descriptions.mnist_citation())

elif data_box == 'Fashion MNIST' and algorithm_box == 'DCGAN':

    z_dim, fashion_mnist_shape, n_classes = 64, (1, 28, 28), 10
    
    generator_input_dim = utils.get_input_dimensions(z_dim, n_classes)
    gen = generators.GeneratorDCGAN_V1_0(input_dim = generator_input_dim).to(device)
    gen.load_state_dict(torch.load('models/gen_conditional_dcgan_fashion_mnist.pth', 
                        map_location = torch.device(device)))
    gen.eval()
    
    col1, col2, col3 = st.columns(spec = [5, 1, 1])

    labels_decoding = ['Camisetas/Tops', 'Calças' ,'Suéteres', 'Vestidos', 'Casacos', 'Sandálias', 
                       'Camisas', 'Tênis', 'Bolsas', 'Botas']

    with col1:
        start_plot_number = st.slider(label = 'Selecione a primeira vestimenta para interpolação:', 
                                      min_value = 0, max_value = 9, value = 1)
        st.markdown(body = f'**Primeira Vestimenta Selecionada**: ```{labels_decoding[start_plot_number]}```')
        end_plot_number = st.slider(label = 'Selecione a segunda vestimenta para interpolação:',
                                    min_value = 0, max_value = 9, value = 9)
        st.markdown(body = f'**Segunda Vestimenta Selecionada**: ```{labels_decoding[end_plot_number]}```')
        n_interpolation = st.slider(label = 'Selecione a quantidade de imagens geradas na interpolação:',
                                    min_value = 10, max_value = 200, value = 100)
    
    site_structure.area_with_interpolation_images(start_plot_number = start_plot_number, 
                                                  end_plot_number = end_plot_number,
                                                  z_dim = z_dim, n_classes = n_classes, device = device, 
                                                  gen = gen, n_interpolation = n_interpolation, column1 = col2, 
                                                  column2 = col3)

    col1, col2, col3 = st.columns(spec = [2, 1, 1])

    with col1:
        n_interpolation = st.slider(label = 'Dígite a quantidade de imagens geradas na interpolação:', 
                                    min_value = 6, max_value = 30, value = 9, step = 3)
        n_noise = st.slider(label = 'Selecione a vestimenta a ser gerada:', min_value = 0, max_value = 9,
                            value = 9)
        st.markdown(body = f'**Vestimenta Selecionada** : ```{labels_decoding[n_noise]}```')
    
    site_structure.area_with_images_generated(n_noise = n_noise, n_classes = n_classes, 
                                              n_interpolation = n_interpolation, z_dim = z_dim, 
                                              gen = gen, device = device, column1 = col2, column2 = col3)
    
    st.markdown(body = '#### Detalhes sobre o Sistema Generativo:')
    st.markdown(body = get_descriptions.dcgan_description())
    st.markdown(body = get_descriptions.dcgan_citation())
    st.markdown(body = '***')
    st.markdown(body = '#### Detalhes sobre a Base de Dados:')
    st.markdown(body = get_descriptions.fashion_mnist_description())
    st.markdown(body = get_descriptions.fashion_mnist_citation())

elif data_box == 'EMNIST Letters' and algorithm_box == 'DCGAN':

    z_dim, mnist_letters_shape, n_classes = 64, (1, 28, 28), 27

    generator_input_dim = utils.get_input_dimensions(z_dim, n_classes)
    gen = generators.GeneratorDCGAN_V1_0(input_dim = generator_input_dim).to(device)
    gen.load_state_dict(torch.load('models/gen_conditional_dcgan_emnist_letters.pth', 
                        map_location = torch.device(device)))
    gen.eval()

    col1, col2, col3 = st.columns(spec = [5, 1, 1])

    labels_decoding = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
                       'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    with col1:
        start_plot_number = st.slider(label = 'Selecione a primeira letra para interpolação:', 
                                      min_value = 1, max_value = 26, value = 15)
        st.markdown(body = f'**Primeira Letra Selecionada**: ```{labels_decoding[start_plot_number - 1]}```')
        end_plot_number = st.slider(label = 'Selecione a segunda letra para interpolação:',
                                    min_value = 1, max_value = 26, value = 11)
        st.markdown(body = f'**Segunda Letra Selecionada**: ```{labels_decoding[end_plot_number - 1]}```')
        n_interpolation = st.slider(label = 'Selecione a quantidade de imagens geradas na interpolação:',
                                    min_value = 10, max_value = 200, value = 100)
    
    site_structure.area_with_interpolation_images(start_plot_number = start_plot_number, 
                                                  end_plot_number = end_plot_number,
                                                  z_dim = z_dim, n_classes = n_classes, device = device, 
                                                  gen = gen, n_interpolation = n_interpolation, column1 = col2, 
                                                  column2 = col3)

    col1, col2, col3 = st.columns(spec = [2, 1, 1])

    with col1:
        n_interpolation = st.slider(label = 'Dígite a quantidade de imagens geradas na interpolação:', 
                                    min_value = 6, max_value = 30, value = 9, step = 3)
        n_noise = st.slider(label = 'Selecione a letra a ser gerada:', min_value = 1, max_value = 26,
                            value = 11)
        st.markdown(body = f'**Letra Selecionada**: ```{labels_decoding[n_noise - 1]}```')
    
    site_structure.area_with_images_generated(n_noise = n_noise, n_classes = n_classes, 
                                              n_interpolation = n_interpolation, z_dim = z_dim, 
                                              gen = gen, device = device, column1 = col2, column2 = col3)
    
    st.markdown(body = '#### Detalhes sobre o Sistema Generativo:')
    st.markdown(body = get_descriptions.dcgan_description())
    st.markdown(body = get_descriptions.dcgan_citation())
    st.markdown(body = '***')
    st.markdown(body = '#### Detalhes sobre a Base de Dados:')
    st.markdown(body = get_descriptions.emnist_letters_description())
    st.markdown(body = get_descriptions.emnist_letters_citation())

elif data_box == 'CIFAR10' and algorithm_box == 'DCGAN':

    z_dim, cifar10_shape, n_classes = 64, (1, 28, 28), 10

    generator_input_dim = utils.get_input_dimensions(z_dim, n_classes)
    gen = generators.GeneratorDCGAN_V1_0(input_dim = generator_input_dim).to(device)
    gen.load_state_dict(torch.load('models/gen_conditional_dcgan_cifar10.pth', 
                        map_location = torch.device(device)))
    gen.eval()

    col1, col2, col3 = st.columns(spec = [5, 1, 1])

    labels_decoding = ['Avião', 'Automóvel', 'Pássaro', 'Gato', 'Cervo', 'Cachorro', 'Sapo', 'Cavalo',
                       'Navio', 'Caminhão']

    with col1:
        start_plot_number = st.slider(label = 'Selecione o primeiro objeto para interpolação:', 
                                      min_value = 0, max_value = 9, value = 8)
        st.markdown(body = f'**Primeiro Objeto Selecionado**: ```{labels_decoding[start_plot_number]}```')
        end_plot_number = st.slider(label = 'Selecione o segundo objeto para interpolação:',
                                    min_value = 0, max_value = 9, value = 7)
        st.markdown(body = f'**Segundo Objeto Selecionado**: ```{labels_decoding[end_plot_number]}```')
        n_interpolation = st.slider(label = 'Selecione a quantidade de imagens geradas na interpolação:',
                                    min_value = 10, max_value = 200, value = 100)
    
    site_structure.area_with_interpolation_images(start_plot_number = start_plot_number, 
                                                  end_plot_number = end_plot_number,
                                                  z_dim = z_dim, n_classes = n_classes, device = device, 
                                                  gen = gen, n_interpolation = n_interpolation, column1 = col2, 
                                                  column2 = col3)

    col1, col2, col3 = st.columns(spec = [2, 1, 1])

    with col1:
        n_interpolation = st.slider(label = 'Dígite a quantidade de imagens geradas na interpolação:', 
                                    min_value = 6, max_value = 30, value = 9, step = 3)
        n_noise = st.slider(label = 'Selecioneo objeto a ser gerado:', min_value = 0, max_value = 9,
                            value = 8)
        st.markdown(body = f'**Objeto Selecionado**: ```{labels_decoding[n_noise]}```')
        
    site_structure.area_with_images_generated(n_noise = n_noise, n_classes = n_classes, 
                                              n_interpolation = n_interpolation, z_dim = z_dim, 
                                              gen = gen, device = device, column1 = col2, column2 = col3)
    
    st.markdown(body = '#### Detalhes sobre o Sistema Generativo:')
    st.markdown(body = get_descriptions.dcgan_description())
    st.markdown(body = get_descriptions.dcgan_citation())
    st.markdown(body = '***')
    st.markdown(body = '#### Detalhes sobre a Base de Dados:')
    st.markdown(body = get_descriptions.cifar10_description())
    st.markdown(body = get_descriptions.cifar10_citation())

else:
    st.warning(body = 'Desculpe, mas essa combinação de base de dados e sistema generativo está indisponível '
                      'no momento.')
    

