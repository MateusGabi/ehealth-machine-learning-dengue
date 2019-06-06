from pandas import *
from sklearn.utils import column_or_1d
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

# RODANDO SVM (SUpport Vector Machine)
def run_SVM(X, y):
	parametrosC = [2**-5, 2**0, 2**5, 2**10]
	parametrosGamma = [2**-15, 2**-10, 2**-5, 2**0,  2**5]

	scoreMedio = 0
	somaScores = 0
	melhorScoreGeral = 0
	melhorCGeral = 0
	melhorGammaGeral = 0

	# para a validação externa utilizaremos 5-fold
	fold_5 = StratifiedKFold(n_splits=5)

	for indices_treino, indices_teste in fold_5.split(X, y):

		# criando novos dados a partir dos indices selecionados
		X_treino = X[indices_treino]
		X_teste = X[indices_teste]
		y_treino = y[indices_treino]
		y_teste = y[indices_teste]

		fold_3 = StratifiedKFold(n_splits=3)

		melhorScore = 0
		melhorGamma = 1
		melhorC = 1

		for indices_treino_2, indices_teste_2 in fold_3.split(X_treino, y_treino):

			X_treino2 = X[indices_treino_2]
			X_teste2 = X[indices_teste_2]
			y_treino2 = y[indices_treino_2]
			y_teste2 = y[indices_teste_2]

			# novos conjuntos de treino e teste criados

			# fazendo grid search nos parametros C e gamma
			for C in parametrosC:
				for gamma in parametrosGamma:
					# inicializando SVM
					svc = SVC(C=C, gamma=gamma)
					# treinando SVM
					svc.fit(X_treino2, y_treino2)
					# medindo acurácia do SVM
					score = svc.score(X_teste2, y_teste2)

					# salvando melhores parametros
					if score > melhorScore:
						melhorScore = score
						melhorGamma = gamma
						melhorC = C

		# treinando novamente SVM, agora utilizando os melhores parametros C e gamma encontrados
		svc = SVC(C=melhorC, gamma = melhorGamma)
		svc.fit(X_treino, y_treino)
		score = svc.score(X_teste, y_teste)
		if score > melhorScoreGeral:
			melhorScoreGeral = score
			melhorGammaGeral = melhorGamma
			melhorCGeral = melhorC

		# acumulando acurácia para cálculo da acurácia média
		somaScores += score

	# calculando e printando acurácia média
	scoreMedio = (1.0 * somaScores) / 5.0
	print("[SVM] Media acuracia do SVM eh: ", scoreMedio)
	print("[SVM] Melhor acuracia alcancada pelo SVM eh: ", melhorScoreGeral, ", Hiperparametros: C= ", melhorCGeral, ", Gamma=", melhorGammaGeral)
	return melhorScoreGeral, scoreMedio


# RODANDO KNN (K Nearest Neighbours)
def run_KNN(X, y):
	parametrosK = [1, 5, 11, 15, 21, 25]

	scoreMedio = 0
	somaScores = 0
	melhorScoreGeral = 0
	melhorKGeral = 0

	# para a validação externa utilizaremos 5-fold
	fold_5 = StratifiedKFold(n_splits=5)

	for indices_treino, indices_teste in fold_5.split(X, y):

		# criando novos dados a partir dos indices selecionados
		X_treino = X[indices_treino]
		X_teste = X[indices_teste]
		y_treino = y[indices_treino]
		y_teste = y[indices_teste]

		fold_3 = StratifiedKFold(n_splits=3)

		melhorScore = 0
		melhorK = 1

		for indices_treino_2, indices_teste_2 in fold_3.split(X_treino, y_treino):

			X_treino2 = X[indices_treino_2]
			X_teste2 = X[indices_teste_2]
			y_treino2 = y[indices_treino_2]
			y_teste2 = y[indices_teste_2]

			# novos conjuntos de treino e teste criados

			# fazendo grid search no parametro K
			for k in parametrosK:
				# inicializando KNN
				knn = KNeighborsClassifier(n_neighbors=k)
				# treinando KNN
				knn.fit(X_treino2, y_treino2)
				# medindo acurácia do KNN
				score = knn.score(X_teste2, y_teste2)

				# salvando melhores parametros
				if score > melhorScore:
					melhorScore = score
					melhorK = k

		# treinando novamente SVM, agora utilizando os melhores parametros C e gamma encontrados
		knn = KNeighborsClassifier(n_neighbors=melhorK)
		knn.fit(X_treino, y_treino)
		score = knn.score(X_teste, y_teste)
		if score > melhorScoreGeral:
			melhorScoreGeral = score
			melhorKGeral = melhorK

		# acumulando acurácia para cálculo da acurácia média
		somaScores += score

	# calculando e printando acurácia média
	scoreMedio = (1.0 * somaScores) / 5.0
	print("[KNN] Media acuracia do KNN eh: ", scoreMedio)
	print("[KNN] Melhor acuracia alcancada pelo KNN eh: ", melhorScoreGeral, ", Hiperparametros: K= ", melhorKGeral)
	return melhorScoreGeral, scoreMedio

# Rodando RF (Random Forest)
def run_RF(X, y):
	nfeaturesList = [10, 15, 20, 25]
	ntreesList = [100, 200, 300, 400]

	scoreMedio = 0
	somaScores = 0
	melhorScoreGeral = 0
	melhorNFeaturesGeral = 0
	melhorNTreesGeral = 0

	# para a validação externa utilizaremos 5-fold
	fold_5 = StratifiedKFold(n_splits=5)

	for indices_treino, indices_teste in fold_5.split(X, y):

		# criando novos dados a partir dos indices selecionados
		X_treino = X[indices_treino]
		X_teste = X[indices_teste]
		y_treino = y[indices_treino]
		y_teste = y[indices_teste]

		fold_3 = StratifiedKFold(n_splits=3)

		melhorScore = 0
		melhorNFeatures = 1
		melhorNTrees = 1

		for indices_treino_2, indices_teste_2 in fold_3.split(X_treino, y_treino):

			X_treino2 = X[indices_treino_2]
			X_teste2 = X[indices_teste_2]
			y_treino2 = y[indices_treino_2]
			y_teste2 = y[indices_teste_2]

			# novos conjuntos de treino e teste criados

			# fazendo grid search nos parametros nFeatures e nTrees
			for nfeatures in nfeaturesList:
				for ntrees in ntreesList:

					# inicializando RF
					rf = RandomForestClassifier(n_estimators=ntrees, max_features=nfeatures)
					# treinando RF
					rf.fit(X_treino2, y_treino2)
					# medindo acurácia do RF
					score = rf.score(X_teste2, y_teste2)

					# salvando melhores parametros
					if score > melhorScore:
						melhorScore = score
						melhorNFeatures = nfeatures
						melhorNTrees = ntrees

		# treinando novamente SVM, agora utilizando os melhores parametros C e gamma encontrados
		rf = RandomForestClassifier(n_estimators=melhorNTrees, max_features=melhorNFeatures)
		rf.fit(X_treino, y_treino)
		score = rf.score(X_teste, y_teste)
		if score > melhorScoreGeral:
			melhorScoreGeral = score
			melhorNTreesGeral = melhorNTrees
			melhorNFeaturesGeral = melhorNFeatures

		# acumulando acurácia para cálculo da acurácia média
		somaScores += score

	# calculando e printando acurácia média
	scoreMedio = (1.0 * somaScores) / 5.0
	print("[RF] Media acuracia RF eh: ", scoreMedio)
	print("[RF] Melhor acuracia alcancada pelo RF eh: ", melhorScoreGeral, ", Hiperparametros: nFeatures= ", melhorNFeaturesGeral, ", nTrees=", melhorNTreesGeral)
	return melhorScoreGeral, scoreMedio


#Rodando GBM (Gradient Boosting Machine)
def run_GBM(X, y):
	learningRateList = [0.1, 0.05]
	ntreesList = [30, 70, 100]

	scoreMedio = 0
	somaScores = 0
	melhorScoreGeral = 0
	melhorLearningRateGeral = 0
	melhorNTreesGeral = 0

	# para a validação externa utilizaremos 5-fold
	fold_5 = StratifiedKFold(n_splits=5)

	for indices_treino, indices_teste in fold_5.split(X, y):

		# criando novos dados a partir dos indices selecionados
		X_treino = X[indices_treino]
		X_teste = X[indices_teste]
		y_treino = y[indices_treino]
		y_teste = y[indices_teste]

		fold_3 = StratifiedKFold(n_splits=3)

		melhorScore = 0
		melhorLearningRate = 1
		melhorNTrees = 1

		for indices_treino_2, indices_teste_2 in fold_3.split(X_treino, y_treino):

			X_treino2 = X[indices_treino_2]
			X_teste2 = X[indices_teste_2]
			y_treino2 = y[indices_treino_2]
			y_teste2 = y[indices_teste_2]

			# novos conjuntos de treino e teste criados

			# fazendo grid search nos parametros learningRate e nTrees
			for learningRate in learningRateList:
				for ntrees in ntreesList:

					# inicializando GBM
					gbm = GradientBoostingClassifier(learning_rate=learningRate, n_estimators=ntrees, max_depth=5)
					# treinando GBM
					gbm.fit(X_treino2, y_treino2)
					# medindo acurácia do GBM
					score = gbm.score(X_teste2, y_teste2)

					# salvando melhores parametros
					if score > melhorScore:
						melhorScore = score
						melhorNTrees = ntrees
						melhorLearningRate = melhorLearningRate

		# treinando novamente SVM, agora utilizando os melhores parametros C e gamma encontrados
		gbm = GradientBoostingClassifier(learning_rate=melhorLearningRate, n_estimators=melhorNTrees, max_depth=5)
		gbm.fit(X_treino, y_treino)
		score = gbm.score(X_teste, y_teste)
		if score > melhorScoreGeral:
			melhorScoreGeral = score
			melhorLearningRateGeral = melhorLearningRate
			melhorNTreesGeral = melhorNTrees

		# acumulando acurácia para cálculo da acurácia média
		somaScores += score

	# calculando e printando acurácia média
	scoreMedio = (1.0 * somaScores) / 5.0
	print("[GBM] Media acuracia GBM eh: ", scoreMedio)
	print("[GBM] Melhor acuracia alcancada pelo GBM eh: ", melhorScoreGeral, ", Hiperparametros: learningRate= ", melhorLearningRateGeral, ", nTrees=", melhorNTreesGeral)
	return melhorScoreGeral, scoreMedio

# Rodando Redes Neurais
def run_Neural(X, y):
	hiddenLayerNeuroniuns = [10, 20, 30, 40]

	scoreMedio = 0
	somaScores = 0
	melhorScoreGeral = 0
	melhorHiddenLayerNGeral = 0

	# para a validação externa utilizaremos 5-fold
	fold_5 = StratifiedKFold(n_splits=5)

	for indices_treino, indices_teste in fold_5.split(X, y):

		# criando novos dados a partir dos indices selecionados
		X_treino = X[indices_treino]
		X_teste = X[indices_teste]
		y_treino = y[indices_treino]
		y_teste = y[indices_teste]

		fold_3 = StratifiedKFold(n_splits=3)

		melhorScore = 0
		melhorHiddenLayerN = 1

		for indices_treino_2, indices_teste_2 in fold_3.split(X_treino, y_treino):

			X_treino2 = X[indices_treino_2]
			X_teste2 = X[indices_teste_2]
			y_treino2 = y[indices_treino_2]
			y_teste2 = y[indices_teste_2]

			# novos conjuntos de treino e teste criados

			# fazendo grid search no parametro hiddenLayerNeuroniuns
			for hiddenLayerN in hiddenLayerNeuroniuns:

				# inicializando rede neural
				neural = MLPClassifier(hidden_layer_sizes=hiddenLayerN, max_iter=300)
				# treinando rede neural
				neural.fit(X_treino2, y_treino2)
				# medindo acurácia da rede neural
				score = neural.score(X_teste2, y_teste2)

				# salvando melhores parametros
				if score > melhorScore:
					melhorScore = score
					melhorHiddenLayerN = hiddenLayerN

		# treinando novamente SVM, agora utilizando os melhores parametros C e gamma encontrados
		neural = MLPClassifier(hidden_layer_sizes=melhorHiddenLayerN, max_iter=300)
		neural.fit(X_treino, y_treino)
		score = neural.score(X_teste, y_teste)
		if score > melhorScoreGeral:
			melhorScoreGeral = score
			melhorHiddenLayerNGeral = melhorHiddenLayerN

		# acumulando acurácia para cálculo da acurácia média
		somaScores += score

	# calculando e printando acurácia média
	scoreMedio = (1.0 * somaScores) / 5.0
	print("[neural] Media acuracia eh: ", scoreMedio)
	print("[neural] Melhor acuracia alcancada pela rede Neural eh: ", melhorScoreGeral, ", Hiperparametros: Numero de Neuronios Camada escondida= ", melhorHiddenLayerNGeral)
	return melhorScoreGeral, scoreMedio


# EXECUTANDO E COMPARANDO OS 5 CLASSIFICADORES

# Lendo CSV utilizando pandas
raw_X = pandas.read_csv('dengue-ml-features.data.csv', sep=',')
raw_Y = pandas.read_csv('dengue-ml-labels.data.csv', sep=',')

# salvando apenas a primeira coluna como classe
y = raw_Y[['total_cases']]

# substituindo dados faltantes pela média da coluna, utilizando Imputer
imp = SimpleImputer()
pre_scale_X = imp.fit_transform(y)

# padronizando as colunas para média 0 e desvio padrão 1, utilizando Scaler
scaler = StandardScaler()
X = scaler.fit_transform(pre_scale_X)

# print(X)
# exit()

maiorScoreSVM, mediaScoreSVM = run_SVM(X,y)	
maiorScoreRF, mediaScoreRF = run_RF(X, y)
maiorScoreGBM, mediaScoreGBM = run_GBM(X, y)
maiorScoreNeural, mediaScoreNeural = run_Neural(X, y)

# guardando valor de X com PCA
# inicializando PCA com 80% de variância
pca = PCA(n_components=0.8)
pca.fit(X)
X_pca = pca.transform(X)
maiorScoreKNN, mediaScoreKNN = run_KNN(X_pca, y)

# descobrindo qual classificador teve melhor acurácia média
if mediaScoreSVM >= mediaScoreRF and mediaScoreSVM >= mediaScoreGBM and mediaScoreSVM >= mediaScoreNeural and mediaScoreSVM >= mediaScoreKNN:
	print("Melhor acuracia media foi do classificador SVM. Acuracia = ", mediaScoreSVM)
if mediaScoreRF >= mediaScoreSVM and mediaScoreRF >= mediaScoreGBM and mediaScoreRF >= mediaScoreNeural and mediaScoreRF >= mediaScoreKNN:
	print("Melhor acuracia media foi do classificador Random Forest. Acuracia = ", mediaScoreRF)
if mediaScoreGBM >= mediaScoreSVM and mediaScoreGBM >= mediaScoreRF and mediaScoreGBM >= mediaScoreNeural and mediaScoreGBM >= mediaScoreKNN:
	print("Melhor acuracia media foi do classificador GradientBoostingMachine. Acuracia = ", mediaScoreGBM)
if mediaScoreNeural >= mediaScoreSVM and mediaScoreNeural >= mediaScoreRF and mediaScoreNeural >= mediaScoreGBM and mediaScoreNeural >= mediaScoreKNN:
	print("Melhor acuracia media foi do classificador de Redes Neurais. Acuracia = ", mediaScoreNeural)
if mediaScoreKNN >= mediaScoreSVM and mediaScoreKNN >= mediaScoreRF and mediaScoreKNN >= mediaScoreGBM and mediaScoreKNN >= mediaScoreNeural:
	print("Melhor acuracia media foi do classificador KNN. Acuracia = ", mediaScoreKNN) 	 		 		

