# pruning-cnn
# 0 - Lancez un run avec salloc et connectez vous sur votre noeud (1 noeud et quelques cpu suffisent) : salloc -p mistral -N 1 --mincpu=20 --time=6:00:00
# 1 - Installez une distribution python : Exécutez le script install_miniconda.sh
# 2 - activez anaconda : source ~/anaconda/bin/activate
# 3 - Créez un environnement dédié au au projet : conda create -y --name pruning python=3.8 pip
# 4 - activez l'environnement : conda activate pruning
# 6 - lancez jupyter-lab : jupyter-lab --ip=0.0.0.0 --NotebookApp.password=''
# 7 - identifiez le port sur lequel il s'est lancé (normalement 8888) ou précisez --port 8888
# 8 - depuis votre machine locale faites un port forwarding pour accéder au port (8888) de votre noeud : ssh -L 8888:sirocco06.formation.cluster:8888 formation
# 9 - connectez vous depuis votre navigateur web sur : 127.0.0.1:8888 (ou votre port du #7)
# 10 - C'est parti !
