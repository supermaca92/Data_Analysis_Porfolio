#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    https://shiny.posit.co/
#

library(shiny)
library(tidyverse)
library(rsconnect)

#####Import Data

dat<-read_csv(url("https://www.dropbox.com/s/uhfstf6g36ghxwp/cces_sample_coursera.csv?raw=1"))
dat<- dat %>% select(c("pid7","ideo5"))
dat<-drop_na(dat)

# Define UI for application 
ui <- fluidPage(

    # Application title
    titlePanel(""),

    # Sidebar with a slider input for number of bins 
    sidebarLayout(
        sidebarPanel(
            sliderInput("my_plot",
                        "Select Five Point Ideology (1=Very liberal, 5=Very conservative)",
                        min = 1,
                        max = 5,
                        value = 5)
        ),

        # Show a plot of the generated distribution
        mainPanel(
           plotOutput("my_plot")
        )
    )
)

# Define server logic required to draw a histogram
server <- function(input, output) {

    output$my_plot <- renderPlot({
      ggplot(filter(dat, 
                    ideo5== input$my_plot),
             aes(x = pid7)) +
        geom_bar() +
        scale_x_continuous(labels = dat$pid7,
                           breaks = dat$pid7) +
        labs(x = "7 Point Party ID, 1 = very D, 7 = very R",
             y = "count") +
        scale_y_continuous(limits = c(0, 125),
                           breaks = seq(0, 125, 25))
    })
}

# Run the application 
shinyApp(ui = ui, server = server)
