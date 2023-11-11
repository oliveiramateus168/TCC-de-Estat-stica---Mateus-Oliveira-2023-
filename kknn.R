# Pacotes
library(tidyverse)
library(tidymodels)
library(MASS)
library(readr)

# Semente aleatoria
set.seed(99, kind = "Mersenne-Twister", normal.kind = "Inversion")


# Dados
dados = read_delim("Heart_Disease_Prediction.csv", 
                    delim = ";", escape_double = FALSE, trim_ws = TRUE)

# Transformando a variável resposta em fator
dados$HeartDisease=as.factor(dados$HeartDisease)

# Separando os dados
dados_split = initial_split(dados, prop = .70, strata = HeartDisease)

# Separando os dados em treino e teste
dados_treino = training(dados_split)
dados_teste  = testing(dados_split)


# Receita
dados_rec =
  recipe(HeartDisease ~ .,
         data = dados_treino) %>%
  step_center(-HeartDisease) %>%
  step_scale(-HeartDisease) %>%
  prep()

# Aplicar a receita nos dados de treino
dados_treino_t = juice(dados_rec)

# Aplicar a receita nos dados de teste
dados_teste_t = bake(dados_rec,
                      new_data = dados_teste)

# Definição do modelo
dados_knn =
  nearest_neighbor() %>%
  set_engine("kknn") %>%
  set_mode("classification")

# Workflow
dados_wflow =
  workflow() %>%
  add_recipe(dados_rec) %>%
  add_model(dados_knn)

# Ajuste do modelo
dados_knn_fit = fit(dados_wflow, dados_treino_t)

# Validação Cruzada
set.seed(99, kind = "Mersenne-Twister", normal.kind = "Inversion")
dados_treino_cv = vfold_cv(dados_treino_t, v = 7)

# Workflow
dados_wflow_cv =
  workflow() %>%
  add_model(dados_knn) %>%
  add_formula(HeartDisease ~ .)

# Rodando validacao cruzada
dados_knn_fit_cv =
  dados_wflow_cv %>%
  fit_resamples(dados_treino_cv)

# Resultados da Validação Cruzada
collect_metrics(dados_knn_fit_cv)

# Resultado final
dados_knn_fit

(dados_knn_fit %>%
    predict(dados_teste_t) %>%
    bind_cols(dados_teste_t) %>%
    metrics(truth = HeartDisease, estimate = .pred_class))


# Tuning

# Definicao do tuning
dados_knn_tune =
  nearest_neighbor(
    neighbors = tune(),
    dist_power = tune()
  ) %>%
  set_engine("kknn") %>%
  set_mode("classification")

dados_knn_tune

# Grid de procura
dados_knn_grid = grid_regular(neighbors(range = c(3,25)),
                                dist_power(range = c(1,2)),
                                levels =c(12, 2))

dados_knn_grid


# Workflow
dados_knn_tune_wflow =
  workflow() %>%
  add_model(dados_knn_tune) %>%
  add_formula(HeartDisease ~ .)

# Avaliacao do modelo
dados_knn_fit_tune =
  dados_knn_tune_wflow %>%
  tune_grid(
    resamples = dados_treino_cv,
    grid = dados_knn_grid
  )


# Resultados
(collect_metrics(dados_knn_fit_tune))

# Melhores modelos
dados_knn_fit_tune %>%
  show_best("accuracy")

# Melhor modelo
dados_knn_best =
  dados_knn_fit_tune %>%
  select_best("accuracy")

dados_knn_final =
  dados_knn_tune_wflow %>%
  finalize_workflow(dados_knn_best)

dados_knn_final = fit(dados_knn_final, dados_treino_t)

dados_knn_final

# Resultados no conjunto de teste
resultado =
  dados_teste_t %>%
  bind_cols(predict(dados_knn_final, dados_teste_t) %>%
              rename(predicao_knn = .pred_class))

metrics(resultado,
        truth = HeartDisease,
        estimate = predicao_knn,
        options = "roc")

# Matriz de confusão 
confusao= table("pred" = resultado$predicao_knn, "obs" = resultado$HeartDisease)
confusao

sensitivity(confusao)
specificity(confusao)


