# Pacotes
library(tidymodels)
library(vip)
library(readr)

# Dados
heartdisease <- read_delim("Heart_Disease_Prediction.csv", 
                                       delim = ";", escape_double = FALSE, trim_ws = TRUE)

heartdisease$HeartDisease = as.factor(heartdisease$HeartDisease) #Transformando a variável resposta em fator

# 70% dos dados como treino
set.seed(99, kind = "Mersenne-Twister", normal.kind = "Inversion")

heartdisease_split = initial_split(heartdisease, prop = .70, strata = HeartDisease)

# Separação do conjunto de treino e teste
heartdisease_treino = training(heartdisease_split)
heartdisease_teste  = testing(heartdisease_split)


# Receita
heartdisease_rec = 
  recipe(HeartDisease ~ ., 
         data = heartdisease_treino) %>%
  step_center(-HeartDisease) %>% 
  step_scale(-HeartDisease) %>% 
  prep()

# Aplicando a receita no conjunto de treino (preparando o conjunto de treino)
heartdisease_treino_t = juice(heartdisease_rec)

# Aplicando a receita no conjunto de treino (preparando o conjunto de teste)
heartdisease_teste_t = bake(heartdisease_rec, new_data = heartdisease_teste)

# Tuning

# Definindo o tuning para os hiperparâmetros
heartdisease_rf_tune =
  rand_forest(
    mtry = tune(), # Número de variáveis preditoras amostradas aleatoriamente em cada divisão
    trees = tune(), # Número de árvores
    min_n = tune() # Número mínimo de amostras para dividir um nó 
  ) %>%
  set_mode("classification") %>%
  set_engine("ranger", importance = "impurity")

# Grid de procura
heartdisease_rf_grid = grid_regular(mtry(range(1, 13)),
                                    min_n(range(50, 100)),
                                    trees(range(500, 2000)),
                                    levels = c(13, 6, 4))

heartdisease_rf_grid
#View(heartdisease_rf_grid)

# workflow
heartdisease_rf_tune_wflow =
  workflow() %>%
  add_model(heartdisease_rf_tune) %>%
  add_formula(HeartDisease ~ .)

# Validação Cruzada
set.seed(99, kind = "Mersenne-Twister", normal.kind = "Inversion")
heartdisease_treino_cv = vfold_cv(heartdisease_treino_t, v = 7)

# Avaliacao do modelo
heartdisease_rf_fit_tune = 
  heartdisease_rf_tune_wflow %>% 
  tune_grid(
    resamples = heartdisease_treino_cv,
    grid = heartdisease_rf_grid
  )

# Resultados
collect_metrics(heartdisease_rf_fit_tune)

# Melhores modelos
heartdisease_rf_fit_tune %>%
  show_best("accuracy")

# Melhor modelo
heartdisease_rf_best = 
  heartdisease_rf_fit_tune %>%
  select_best("accuracy")

heartdisease_rf_final =
  heartdisease_rf_tune_wflow %>%
  finalize_workflow(heartdisease_rf_best)

heartdisease_rf_final = fit(heartdisease_rf_final, 
                             heartdisease_treino_t)

heartdisease_rf_final

# Resultados no conjunto de teste
resultado_rf = 
  heartdisease_teste_t %>%
  bind_cols(predict(heartdisease_rf_final, heartdisease_teste_t) %>%
              rename(predicao_rf = .pred_class))

metrics(resultado_rf, 
        truth = HeartDisease, 
        estimate = predicao_rf,
        options = "acuracy")

# Matriz de confusão
conf_mat(resultado_rf, 
         truth = HeartDisease, 
         estimate = predicao_rf) %>%
  autoplot(type = "heatmap")
table(resultado_rf$HeartDisease, resultado_rf$predicao_rf)
confusao= table("pred" = resultado_rf$predicao_rf, "obs" = resultado_rf$HeartDisease)
confusao
sensitivity(confusao)
specificity(confusao)

# Importancia das variaveis
heartdisease_rf_final %>% 
  pull_workflow_fit() %>% 
  vip(scale = TRUE)



