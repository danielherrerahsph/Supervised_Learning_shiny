library(shinythemes)
library(shiny)
library(tidyverse)
library(ggthemes)
library(ggrepel)
library(gridExtra)
library(caret)



binary_methods <- c("Logistic regression", "K-nearest neighbors", "Decision Trees")

#############################
ui <- fluidPage(theme = shinytheme("sandstone"),
                titlePanel("Intro to Machine Learning"),
                
                # create first tab 
                tabsetPanel(
                  # name the tab
                  tabPanel("Initial Exploratory Data Analysis",
                           sidebarLayout(
                             sidebarPanel(
                               p(strong("Input your data in the following order:")),
                               p(strong("Outcome variable (Y), then all other predictor variables")),
                               p("Y, X1,X2,X3....Xn"),
                               p(strong("Include headers")),
                               # accept only csv
                               fileInput("upload", NULL, buttonLabel = "Upload Your Data as .csv", multiple = FALSE, accept = ".csv"),
                               br(),
                               p("The plots to the right are examples to compare your data output. 
                                       Determine which plot is most similar to your own plot output and choose the appropriate tab for your next step.")),
                             
                             
                             
                             
                             
                             mainPanel(
                               plotOutput(outputId = "samp_plot"),
                               # eventually add another plot where which will be histogram using inputed data (get y only)
                               plotOutput("contents")
                             )
                           )
                  ),
                  
                  # maybe add model statment printout
                  tabPanel("Binary Classification",
                           sidebarLayout(
                             sidebarPanel(
                               p(strong("What type of binary classifcation would you like to do?")),
                               selectInput("method", "What method would you like to use?", binary_methods)),
                             
                             
                             mainPanel(
                               tableOutput("summary")
                             )
                           )
                  )
                )
                
)

server <- function(input, output, session) {
  output$samp_plot <- renderPlot({
    # establish the potential outcome plots
    norm_ydata  <- data.frame( y = rnorm(100, 0, 1))
    binary_data <- data.frame( y = rbinom(n = 100, prob = .6, size = 1))
    multi_data <- data.frame( y = rbinom(n = 100, prob = .6, size = 4))
    
    
    # create the three plots
    # normal plot histogram for continuous outcome
    norm_plot <- 
      ggplot(aes(x = y), data = norm_ydata) + 
      geom_histogram(binwidth =0.5, fill = "lightsteelblue3", col = "black") + 
      ylab("Frequency") +
      ggtitle("Continuous Outcome Y Example") + 
      theme_bw() + 
      theme(axis.title.x=element_blank(), 
            panel.grid.major.x = element_blank(),
            panel.grid.minor.x = element_blank())
    
    
    # histogram if outcome is binary
    binary_plot <- 
      ggplot(aes(x = y), data = binary_data) +
      geom_histogram(binwidth =0.5, fill = "lightsteelblue3", col = "black") + 
      ggtitle("Binary Outcome Y Example") + 
      theme_bw() + 
      theme(axis.title.x=element_blank(),
            axis.title.y = element_blank(), 
            panel.grid.major.x = element_blank(),
            panel.grid.minor.x = element_blank())
    
    
    
    # histogram if outcome is ordinal/multinomial
    multinomial_plot <- 
      ggplot(aes(x = y), data = multi_data) +
      geom_histogram(binwidth =0.5, fill = "lightsteelblue3", col = "black") + 
      ggtitle("Multinomial or Ordinal Outcome Y Example") + 
      theme_bw() + 
      theme(axis.title.x=element_blank(),
            axis.title.y = element_blank(),
            panel.grid.major.x = element_blank(), # Horizontal major grid lines
            panel.grid.minor.x = element_blank())
    
    
    grid.arrange(norm_plot, binary_plot, multinomial_plot, ncol = 3)
    
  }
  )
  
  # here we manipulate the input data 
  output$contents <- renderPlot({
    file <- input$upload
    ext <- tools::file_ext(file$datapath)
    
    req(file)
    # check csv 
    validate(need(ext == "csv", "Please upload a csv file"))
    
    mydata <- read.csv(file = file$datapath, header = TRUE)
    
    # rename variables
    # need col 1 to be outcome
    # create x1,x2...xn for predictors
    headers <- sprintf("X%d",seq(1:(length(mydata)-1)))
    # new names 
    colnames(mydata) <- c("y", headers)
    # need to find a way to rename the variables so i can access it as y or index appropriate using base [,1]
    ggplot(aes(x= y), data = mydata) +
      geom_histogram(binwidth =0.5, fill = "lightsteelblue3", col = "black") + 
      xlab("Predictor Variable") +
      ggtitle("Outcome Variable from Your Data") + 
      theme_bw()
  })
  
  # summary table for tab 2 binary outcome
  output$summary <- renderTable({
    # must recreate dataset, seems inefficient
    file <- input$upload
    ext <- tools::file_ext(file$datapath)
    
    req(file)
    # check csv 
    validate(need(ext == "csv", "Please upload a csv file"))
    
    mydata <- read.csv(file = file$datapath, header = TRUE)
    
    # rename variables
    # need col 1 to be outcome
    # create x1,x2...xn for predictors
    headers <- sprintf("X%d",seq(1:(length(mydata)-1)))
    # new names 
    colnames(mydata) <- c("y", headers)
    
    # do logistic regression
    if (input$method == "Logistic regression"){
      # maybe move outside this function
      set.seed(1)
      # split test and training data into 70.30 split
      trainIndex <- createDataPartition(mydata$y, p = .7, 
                                        list = FALSE, 
                                        times = 1)
      train_set <- mydata[ trainIndex,]
      test_set <- mydata[-trainIndex,]
      # fit and predict
      logistic_model <- glm(y ~ ., data = train_set, family = "binomial")
      glm_probs <- predict(logistic_model, newdata = test_set, type = "response")
      glm_preds <- ifelse(glm_probs > 0.5, 1, 0)
      
      accuracy <- confusionMatrix(data = as.factor(glm_preds), reference = as.factor(test_set$y), positive = "1")$overall[1]
      metrics <-  confusionMatrix(data = as.factor(glm_preds), reference = as.factor(test_set$y), positive = "1")$byClass[1:4]
      glm_data <- data.frame("accuracy" = accuracy, 
                             "sensitivity" = metrics[1],
                             "specificity" = metrics[2],
                             "PPV" = metrics[3],
                             "NPV" = metrics[4])
      glm_data
    }
    
    # k nearest neighbors, fo rnow with default k = 10
    else if (input$method == "K-nearest neighbors"){
      # maybe move outside this function
      set.seed(1)
      trainIndex <- createDataPartition(mydata$y, p = .7, 
                                        list = FALSE, 
                                        times = 1)
      train_set <- mydata[ trainIndex,]
      test_set <- mydata[-trainIndex,]
      knn_mod <- knn3(y ~ ., data = train_set, k = 10)
      knn_probs <- predict(knn_mod, newdata = test_set)[,2]
      knn_preds <- ifelse(knn_probs > 0.5, 1, 0)
      
      accuracy_knn <- confusionMatrix(data = as.factor(knn_preds), reference = as.factor(test_set$y), positive = "1")$overall[1]
      metrics_knn <-  confusionMatrix(data = as.factor(knn_preds), reference = as.factor(test_set$y), positive = "1")$byClass[1:4]
      knn_data <- data.frame("accuracy" = accuracy_knn, 
                             "sensitivity" = metrics_knn[1],
                             "specificity" = metrics_knn[2],
                             "PPV" = metrics_knn[3],
                             "NPV" = metrics_knn[4])
      knn_data
    }
    
  }
  
  )
  
}

# Run the app ----
shinyApp(ui = ui, server = server)





