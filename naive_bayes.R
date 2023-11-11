#Naive Bayes usando o Tidymodels

#Pacotes
library(tidymodels)
library(readr)
library(ggfortify)
library(klaR)
library(discrim)
library(patchwork)

#Dados (variável categórica como fator)
heartdisease = read_delim("Heart_Disease_Prediction.csv", delim = ";", escape_double = FALSE, trim_ws = TRUE)
heartdisease$HeartDisease = as.factor(heartdisease$HeartDisease) #Transformando a variável resposta em fator
heartdisease$HeartDisease = as.factor(heartdisease$HeartDisease)
heartdisease$Sex = as.factor(heartdisease$Sex)
heartdisease$Chest_pain_type = as.factor(heartdisease$Chest_pain_type)
heartdisease$BP = as.factor(heartdisease$BP)
heartdisease$Cholesterol = as.factor(heartdisease$Cholesterol)
heartdisease$Exercise_angina = as.factor(heartdisease$Exercise_angina)
heartdisease$Slope_of_ST = as.factor(heartdisease$Slope_of_ST)
heartdisease$Number_of_vessels_fluro = as.factor(heartdisease$Number_of_vessels_fluro)
heartdisease$Thallium = as.factor(heartdisease$Thallium)

# Treino e teste (para variáveis categóricas como fator)
set.seed(99, kind = "Mersenne-Twister", normal.kind = "Inversion")
heartdisease_split = initial_split(heartdisease, prop = .70, strata = HeartDisease)
treino = training(heartdisease_split)
teste = testing(heartdisease_split)

# Receita (para variáveis categóricas como fator) 
heartdisease_rec = 
  recipe(HeartDisease ~ ., 
         data = treino) %>%
  step_center(-c("HeartDisease","Sex","Chest_pain_type","BP","Cholesterol","Exercise_angina","Slope_of_ST","Number_of_vessels_fluro","Thallium")) %>%  
  step_scale(-c("HeartDisease","Sex","Chest_pain_type","BP","Cholesterol","Exercise_angina","Slope_of_ST","Number_of_vessels_fluro","Thallium")) %>%
  prep()

# Aplicando a receita no conjunto de treino (para variáveis categóricas como fator)
treino_t = juice(heartdisease_rec); treino_t

# Aplicando a receita no conjunto de teste (para variáveis categóricas como fator)
teste_t = bake(heartdisease_rec, new_data = teste); teste_t

# Validação cruzada
set.seed(99, kind = "Mersenne-Twister", normal.kind = "Inversion")
treino_validacao_cruzada = vfold_cv(treino_t, v = 7)

# Definição do modelo
treino_bayes = naive_Bayes(
  mode = "classification",
  engine = "klaR"
)

# Workflow
treino_wflow = 
  workflow() %>% 
  add_recipe(heartdisease_rec) %>%
  add_model(treino_bayes)

# Ajuste do modelo
treino_bayes_fit = fit(treino_wflow, treino_t)
head(treino_bayes_fit)

# Workflow da validação cruzada
treino_wflow_validacao_cruzada = 
  workflow() %>% 
  add_model(treino_bayes) %>%
  add_formula(HeartDisease ~ .)

# Aplicando a validação cruzada
treino_bayes_fit_validacao_cruzada = 
  treino_wflow_validacao_cruzada %>%
  fit_resamples(treino_validacao_cruzada); treino_bayes_fit_validacao_cruzada


# Resultados da validação cruzada no conjunto de treino
collect_metrics(treino_bayes_fit_validacao_cruzada)

# Resultado final
head(treino_bayes_fit)

# No conjunto de teste
treino_bayes_fit %>% 
  predict(teste_t) %>% 
  bind_cols(teste_t) %>% 
  metrics(truth = HeartDisease, estimate = .pred_class)

# Resumo dos resultados
# Treino
collect_metrics(treino_bayes_fit_validacao_cruzada)[1, 3]

# Teste
(treino_bayes_fit %>% 
    predict(teste_t) %>% 
    bind_cols(teste_t) %>% 
    metrics(truth = HeartDisease, estimate = .pred_class))[1, 3]

# Resultado da predição no conjunto de teste
resultado = 
  teste_t %>%
  bind_cols(predict(treino_bayes_fit, teste_t) %>%
              rename(predicao_bayes = .pred_class))
#View(resultado)

# Matriz de confusão
table(resultado$HeartDisease, resultado$predicao_bayes)



#Tuning

# definicao do tunning
heartdisease_bayes_tune =
  naive_Bayes(
    Laplace = tune()
  )

heartdisease_bayes_tune


# Grid de procura
heartdisease_bayes_grid = grid_regular(Laplace(range = c(0,2)),
                                levels = c(5))

heartdisease_bayes_grid

# Workflow
heartdisease_bayes_tune_wflow =
  workflow() %>% 
  add_model(heartdisease_bayes_tune) %>% 
  add_formula(HeartDisease ~ .)

# Avaliacao do modelo
heartdisease_bayes_fit_tune = 
  heartdisease_bayes_tune_wflow %>% 
  tune_grid(
    resamples = treino_validacao_cruzada, #
    grid = heartdisease_bayes_grid
  )
heartdisease_bayes_fit_tune

# Resultados
collect_metrics(heartdisease_bayes_fit_tune)

# Melhores modelos
heartdisease_bayes_fit_tune %>%
  show_best("accuracy")

# melhor modelo
heartdisease_bayes_best = 
  heartdisease_bayes_fit_tune %>%
  select_best("accuracy")

heartdisease_bayes_final =
  heartdisease_bayes_tune_wflow %>%
  finalize_workflow(heartdisease_bayes_best)

heartdisease_bayes_final <- fit(heartdisease_bayes_final, treino_t)

heartdisease_bayes_final

# Resultados no conjunto de teste
(resultado = 
  teste_t %>%
  bind_cols(predict(heartdisease_bayes_final, teste_t) %>%
              rename(predicao_bayes = .pred_class)))

metrics(resultado, 
        truth = HeartDisease, 
        estimate = predicao_bayes)

(confusaoBayes = table(resultado$predicao_bayes, resultado$HeartDisease))

sensitivity(confusaoBayes)
specificity(confusaoBayes)

