#!/bin/sh

if [ $CONDA_DEFAULT_ENV = "nlp" ]; then
    echo "Instalando dependências adicionais..."
    # instalando pacotes para possibilitar treinamento
    pip install spacy-lookups-data

    # instalando modelos em inglês e português
    python -m spacy download en_core_web_sm
    python -m spacy download pt_core_news_sm
else
    echo "Ambiente errado! Execute: conda activate nlp"
fi