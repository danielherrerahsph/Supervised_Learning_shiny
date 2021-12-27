library(shinythemes)
library(shinyWidgets)
library(shiny)
library(rpart)
library(tidyverse)
library(ggthemes)
library(ggrepel)
library(gridExtra)
library(caret)
library(e1071)
library(pROC)



binary_methods <- c("Logistic regression", "K-nearest neighbors", "Naive Bayes")

#############################
ui <- fluidPage(theme = shinytheme("sandstone"),
                titlePanel("Supervised Learning: Classification"),
                
                 setBackgroundImage(
                  src = "https://img.rawpixel.com/s3fs-private/rawpixel_images/website_content/rm283-nunny-030_1.jpg?w=800&dpr=1&fit=default&crop=default&q=65&vib=3&con=3&usm=15&bg=F4F4F3&auto=format&ixlib=js-2.2.1&s=20dc84128067233e1b7bb21f44a7396c"
                 ),
                
                # first tab will be for data upload
                tabsetPanel(
                    # name the tab
                    tabPanel("Data Upload",
                             sidebarLayout(
                                 sidebarPanel(
                                     p(strong("Upload your dataset below (headers included):")),
                                     p(strong("Then select your outcome variable. Make sure it is binary (1 = yes, 0 = no).")),
                                     br(),
                                     # accept only csv
                                     fileInput("upload", NULL, buttonLabel = "Upload Your Data as .csv", multiple = FALSE, accept = ".csv"),
                                     
                                     br(),
                                     
                                     p("The plot at the top is an example to compare with your plot output. Ensure a binary random variable is chosen to proceed to classification supervised learning algorithms."), 
                                     
                                     br(),
                                     
                                     varSelectInput("outcome", "Select your outcome variable:", data = data.frame())),
                                 
                                 
                                 
                                 
                                 
                                 
                                 mainPanel(
                                     plotOutput(outputId = "samp_plot", width = 500),
                                     # eventually add another plot where which will be histogram using inputed data (get y only)
                                     plotOutput("contents", width  = 500)                             )
                             )
                    ),
                    
                    
                    # Second panel is logistic regression
                    tabPanel("Logistic Regression",
                             sidebarLayout(
                                 sidebarPanel(
                                     p("Click to perform logistic regression"),
                                     actionButton("logistic", "Perform")),
                                 
                                 
                                 
                                 mainPanel(
                                     tableOutput("summary_log"),
                                     plotOutput("myroc_log", width = 600)
                                 )
                             )
                    ),
                    
                    
                    #third panel -knn
                    tabPanel("K-Nearest Neighbors",
                             sidebarLayout(
                                 sidebarPanel(
                                     sliderInput("neighbors", "Select neighbors (k):", min = 1, max = 20, value = 2),
                                     br(),
                                     p("Click to perform K-Nearest Neighbors"),
                                     actionButton("knn", "Perform")),
                                 
                                 
                                 
                                 mainPanel(
                                     tableOutput("summary_knn"),
                                     plotOutput("myroc_knn", width = 600)
                                 )
                             )
                    ),
                    
                    
                    
                    #forth panel -decision trees
                    tabPanel("Decision Tree",
                             sidebarLayout(
                                 sidebarPanel(
                                     numericInput("cp", "Select complexity parameter (cp):", min = 0.001, max = 5, value = .01, step = .001),
                                     br(),
                                     p("Click to create a decision tree"),
                                     actionButton("dt", "Create")),
                                 
                                 
                                 
                                 mainPanel(
                                     tableOutput("summary_dt"),
                                     plotOutput("myroc_dt", width = 600)
                                 )
                             )
                    )
                    
                    
                )
                
)


server <- function(input, output, session) {
    

# tab 1 outputs
    # add functionality to select outcome variable
    observeEvent(input$upload, {
        
        mytable <- read.csv(input$upload$datapath)
        
        updateVarSelectInput(session, "outcome", label = "Select", data = mytable)
        
    })
    
    output$samp_plot <- renderPlot({
        # establish the potential outcome plot
        binary_data <- data.frame( y = rbinom(n = 100, prob = .6, size = 1))
        
        
        
        # histogram if outcome is binary
        ggplot(aes(x = y), data = binary_data) +
            geom_histogram(binwidth =0.5, fill = "lightsteelblue3", col = "black") +
            scale_x_continuous(breaks = c(0,1)) + 
            ggtitle("Binary Outcome Example") + 
            theme_bw() + 
            theme(axis.title.x=element_blank(),
                  axis.title.y = element_blank(), 
                  panel.grid.major.x = element_blank(),
                  panel.grid.minor.x = element_blank())
        
        
    }
    )
    
    
    output$contents <- renderPlot({
        
        # must recreate dataset, seems inefficient
        file <- input$upload
        ext <- tools::file_ext(file$datapath)
        
        req(file)
        # check csv 
        validate(need(ext == "csv", "Please upload a csv file"))
        
        mydata <- read.csv(file = file$datapath, header = TRUE)
        
        # make outcome variable be named y for simplicity
        mydata <- mydata %>% 
            rename(., y = !!input$outcome)
        
        
        
        ggplot(mydata, aes(x = factor(y))) +
            stat_count(fill = "dodgerblue4", col = "black", stat = "count") +
            xlab("Predictor Variable") +
            ggtitle("Outcome Variable from Your Data") + 
            theme_bw() + 
            theme(axis.title.x=element_blank(), 
                  panel.grid.major.x = element_blank(),
                  panel.grid.minor.x = element_blank(),
                  )
        
    })
    
    
    
    
# tab 2 outputs - logistic  
    # summary table for tab 2 binary outcome
    observeEvent(input$logistic, {
        
        # must recreate dataset, seems inefficient
        file <- input$upload
        ext <- tools::file_ext(file$datapath)
        
        req(file)
        # check csv 
        validate(need(ext == "csv", "Please upload a csv file"))
        
        mydata <- read.csv(file = file$datapath, header = TRUE)
        
        # make outcome variable be named y for simplicity
        mydata <- mydata %>% 
            rename(., y = !!input$outcome)
        
        output$summary_log <- renderTable({
            
            
            # do logistic regression
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
        })
       
        output$myroc_log <- renderPlot({
            # must recreate, which seems highly inefficient
            set.seed(1)
            
            # split test and training data into 70.30 split
            trainIndex <- createDataPartition(mydata$y, p = .7, 
                                              list = FALSE, 
                                              times = 1)
            train_set <- mydata[ trainIndex,]
            test_set <- mydata[-trainIndex,]
            
            logistic_model <- glm(y ~ ., data = train_set, family = "binomial")
            glm_probs <- predict(logistic_model, newdata = test_set, type = "response")
            glm_preds <- ifelse(glm_probs > 0.5, 1, 0)
            
            ggroc(roc(test_set$y, glm_probs), legacy.axes = T, col = "lightsteelblue3", size = 2) +
                geom_abline(linetype = "dashed", alpha = 0.4) + 
                ggtitle(paste0("ROC Curve")) +
                xlab("1- Specificity") + 
                ylab("Sensitivity")+
                guides(colour = guide_legend(title = "Models")) + 
                theme_bw() +
                theme(title = element_text(size = 15),
                      axis.title = element_text(size = 12, face = "bold"),
                      legend.title = element_text(size = 10, face = "bold"),
                      panel.border = element_blank())

            
            
        }) 
        
    })
    
    
    
    
# tab 3 outputs - knn

    observeEvent(input$knn, {
        output$summary_knn <- renderTable({
            # must recreate dataset, seems inefficient
            file <- input$upload
            ext <- tools::file_ext(file$datapath)
            
            req(file)
            # check csv 
            validate(need(ext == "csv", "Please upload a csv file"))
            
            mydata <- read.csv(file = file$datapath, header = TRUE)
            
            # make outcome variable be named y for simplicity
            mydata <- mydata %>% 
                rename(., y = !!input$outcome)
            
            
            # do knn 
            # maybe move outside this function
            set.seed(1)
            
            # split test and training data into 70.30 split
            trainIndex <- createDataPartition(mydata$y, p = .7, 
                                              list = FALSE, 
                                              times = 1)
            train_set <- mydata[ trainIndex,]
            test_set <- mydata[-trainIndex,]
            # fit and predict
            knn_mod <- knn3(y ~ ., data = train_set, k = as.numeric(input$neighbors))
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
        })
        
        output$myroc_knn <- renderPlot({
            
            ggroc(roc(test_set$y, knn_probs), legacy.axes = T, col = "lightsteelblue3", size = 2) +
                geom_abline(linetype = "dashed", alpha = 0.4) + 
                ggtitle(paste0("ROC Curve (k = ",input$neighbors, ")")) +
                xlab("1- Specificity") + 
                ylab("Sensitivity")+
                guides(colour = guide_legend(title = "Models")) + 
                theme_bw() +
                theme(title = element_text(size = 15),
                      axis.title = element_text(size = 12, face = "bold"),
                      legend.title = element_text(size = 10, face = "bold"),
                      panel.border = element_blank())
            
            
            
        }) 
        
    })
    
    
    
    
# tab 4 - decision trees
    
    
    observeEvent(input$dt, {
        output$summary_dt <- renderTable({
            
            # must recreate dataset, seems inefficient
            file <- input$upload
            ext <- tools::file_ext(file$datapath)
            
            req(file)
            # check csv 
            validate(need(ext == "csv", "Please upload a csv file"))
            
            mydata <- read.csv(file = file$datapath, header = TRUE)
            
            # make outcome variable be named y for simplicity
            mydata <- mydata %>% 
                rename(., y = !!input$outcome)
            
            # maybe move outside this function
            set.seed(1)
            
            # split test and training data into 70.30 split
            trainIndex <- createDataPartition(mydata$y, p = .7, 
                                              list = FALSE, 
                                              times = 1)
            train_set <- mydata[ trainIndex,]
            test_set <- mydata[-trainIndex,]
            
            # fit and predict
            tree_model <- rpart(y ~ ., data = train_set, cp = input$cp)
            tree_probs <- predict(tree_model, newdata = test_set)
            
            tree_preds <- factor(ifelse(tree_probs >= 0.5, 1, 0))
            
            accuracy_dt <- confusionMatrix(data = as.factor(tree_preds), reference = as.factor(test_set$y), positive = "1")$overall[1]
            metrics_dt <-  confusionMatrix(data = as.factor(tree_preds), reference = as.factor(test_set$y), positive = "1")$byClass[1:4]
            dt_data <- data.frame("accuracy" = accuracy_dt, 
                                   "sensitivity" = metrics_dt[1],
                                   "specificity" = metrics_dt[2],
                                   "PPV" = metrics_dt[3],
                                   "NPV" = metrics_dt[4])
            dt_data
        })
        
        output$myroc_dt <- renderPlot({
            
            ggroc(roc(test_set$y, dt_probs), legacy.axes = T, col = "lightsteelblue3", size = 2) +
                geom_abline(linetype = "dashed", alpha = 0.4) + 
                ggtitle(paste0("ROC Curve (k = ",input$neighbors, ")")) +
                xlab("1- Specificity") + 
                ylab("Sensitivity")+
                guides(colour = guide_legend(title = "Models")) + 
                theme_bw() +
                theme(title = element_text(size = 15),
                      axis.title = element_text(size = 12, face = "bold"),
                      legend.title = element_text(size = 10, face = "bold"),
                      panel.border = element_blank())
            
            
            
        }) 
        
    })
    
    
}


# Run the app ----
shinyApp(ui = ui, server = server)
