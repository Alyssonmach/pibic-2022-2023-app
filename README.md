# Geração de Imagens Artificiais com GANs

Aplicação Streamlit desenvolvida como parte de uma pesquisa de iniciação científica (PIBIC) na Universidade Federal de Campina Grande. O projeto demonstra a geração e a interpolação de imagens com diferentes arquiteturas de redes adversárias generativas.

## Executar localmente

O ambiente registrado no projeto usa Python 3.8.10 (`runtime.txt`) e as dependências fixadas em `requirements.txt`.

Essas versões preservam o ambiente original da pesquisa. Python 3.8 encerrou o suporte em 7 de outubro de 2024; antes de usar o projeto em uma implantação nova, planeje a atualização das dependências e valide a geração dos modelos. Consulte o [status oficial das versões do Python](https://devguide.python.org/versions/).

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run main.py
```

Depois, abra o endereço local mostrado pelo Streamlit no terminal. O app carrega os pesos dos geradores em `models/generators/` e executa a geração na CPU. Os conjuntos de dados não precisam ser baixados para usar a interface. Os arquivos em `models/discriminators/` são artefatos de treinamento e não são carregados pela interface atual.

## Estrutura

- `main.py`: interface e seleção de arquitetura e conjunto de imagens.
- `generators.py`: arquiteturas dos geradores.
- `site_structure.py` e `get_descriptions.py`: componentes e conteúdo exibidos no app.
- `utils.py`: geração, interpolação e gravação de imagens.
- `models/generators/`: pesos usados pela interface.
- `models/discriminators/`: pesos dos discriminadores e críticos salvos durante os experimentos.
- `images/` e `data/`: imagens de demonstração versionadas.

As imagens produzidas durante uma sessão são gravadas em `.generated/`, que é ignorada pelo Git. Assim, a execução do app não altera as imagens de demonstração versionadas em `images/`.

## GitHub e versões

Os pesos dos modelos são mantidos no repositório porque a interface depende deles. Atualmente, eles ocupam cerca de 113 MiB no total e o maior arquivo tem aproximadamente 13,3 MB. Isso está abaixo do limite de 100 MiB por arquivo do GitHub; Git LFS não é necessário neste momento. Se novos pesos crescerem bastante, considere publicá-los em uma Release ou usar Git LFS antes de adicioná-los ao histórico. Consulte [a documentação do GitHub sobre arquivos grandes](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

Use `dev` para o trabalho em andamento e promova alterações revisadas para `main`. Para marcar uma versão publicada, crie uma tag, por exemplo `v1.0.0`; não é necessário criar tags para cada commit.

## Licença

O código deste projeto está sob a licença MIT. Consulte [LICENSE](LICENSE).
