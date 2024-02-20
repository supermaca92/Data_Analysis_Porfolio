
library(shiny)
library(tidyverse)
library(plotly)
library(DT)

#####Import Data

dat<-read_csv(url("https://www.dropbox.com/s/uhfstf6g36ghxwp/cces_sample_coursera.csv?raw=1"))
dat<- dat %>% select(c("pid7","ideo5","newsint","gender","educ","CC18_308a","region"))
dat<-drop_na(dat)

# Define UI for page 1

ui <- fluidPage(
  
  # Set title
  
  navbarPage( title = "My Application",
              fluid = TRUE,
              
              # Page 1
              
              tabPanel("Page 1",
                       fluidRow(
                         
                         # Column 1 = slider
                         
                         column(4,
                                inputPanel(sliderInput(inputId = "slider1",
                                                       label = "Select Five Point Ideology (1=Very liberal, 5=Very conservative)", 
                                                       min = 0, 
                                                       max = 5, 
                                                       sep = "", 
                                                       value = 5)
                                )),
                         # Column 2 = Figures
                         
                         column(8,
                                mainPanel(
                                  tabsetPanel(
                                    tabPanel("Tab 1", plotOutput("plot1")), 
                                    tabPanel("Tab 2", plotOutput("plot2"))
                                  ))),)),
              # Page 2
              
              tabPanel("Page 2",
                       fluidRow(
                         
                         # Column 1 = Checker box             
                         column(1,
                                inputPanel(checkboxGroupInput(inputId = "slider2",
                                                              label = "Gender",
                                                              c("1", 
                                                                "2")
                                )))),
                       # Column 2 = Plot   
                       column(11,
                              plotlyOutput("plot3"))),

                         #Page 3
                         
                         tabPanel("Page 3",
                                  fluidRow(
                                    column(4,
                                           inputPanel(selectInput(inputId = "slicer3",
                                                                  label = "Region",
                                                                  c(1, 2, 3, 4)
                                                                  ))),
                                    column(8,
                                           dataTableOutput("table1"))
                                  )
                                  )))


# Define server logic required to draw a histogram
server <- function(input, output) {
  
  output$plot1 <- renderPlot({
    ggplot(filter(dat, 
                  ideo5 == input$slider1),
           aes(x = pid7)) +
      geom_bar() +
      scale_x_continuous(labels = dat$pid7,
                         breaks = dat$pid7) +
      labs(x = "7 Point Party ID, 1 = very D, 7 = very R",
           y = "count") +
      scale_y_continuous(limits = c(0, 125),
                         breaks = seq(0, 125, 25))
  })
  
  output$plot2 <- renderPlot({
    ggplot(filter(dat,
                  ideo5 == input$slider1),
           aes(x = CC18_308a)) +
      geom_bar() +
      scale_x_continuous(labels = dat$CC18_308a,
                         breaks = dat$CC18_308a) +
      scale_y_continuous(limits = c(0, 150),
                         breaks = seq(0, 150, 50)) +
      labs(x = "Trump Support",
           y = "count")
  })
  
  output$plot3 <- renderPlotly({
     ggplotly(
       ggplot(filter(dat,
                  gender %in% input$slider2),
           aes(x = educ,
               y = pid7)) +
      geom_point(position=position_jitter(h=0.1, w=0.1)) +
      geom_smooth(method = "lm")
     )  
  })
  
  output$table1 <- renderDataTable({
    datatable(filter(dat,
                     region %in% input$slicer3),
              options = list(lengthMenu = c(10, 30, 50)))
  })
  
}    


# Run the application 
shinyApp(ui = ui, server = server)
