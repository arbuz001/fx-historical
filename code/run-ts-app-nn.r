# http://stats.stackexchange.com/questions/10162/how-to-apply-neural-network-to-time-series-forecasting
# source("c:/alexey_workspace/sauder/kaggle/data-fx-historical/code/code.R")
library(MASS)
library(neuralnet)

fnScaleData = function(z){

	z.scaled = (z - mean(z))/sd(z)
	return(z.scaled)
}

# read data from the input file
strPrjPath	= "c:/works-and-documents/svn/kaggle/data-fx-historical/"
strDataPath	= paste(strPrjPath,"data/", sep = "")
strOut		= paste(strPrjPath,"out/", sep = "")
strFileIn	= paste(strDataPath,"USD.RUB.trimmed.csv", sep = "")

# dataIn	= fnExtractDataFromCSV(strFileIn)
dataIn = read.table(strFileIn, header = TRUE, sep = ",")

full.z = ts(dataIn["RUB.USD"])
plot(full.z)
abline(reg = lm(full.z ~ time(full.z)))

# full.z.scaled = fnScaleData(full.z)
full.z.scaled = full.z

# gLogR = log(full.z.scaled[1:n-1]/full.z.scaled [2:n])
gLogR	= log(full.z.scaled)
nN		= length(gLogR)

gLogRDelta = gLogR[2:nN] - gLogR[1:nN-1]

rWindowSize	= 0.3
nWindowSize	= floor(nN*rWindowSize)

gLogRDeltaWindowed	= gLogRDelta[1:nWindowSize]

cov(gLogRDeltaWindowed[1:nWindowSize-1],gLogRDeltaWindowed[2:nWindowSize]) 

acf(x, lag.max = NULL,
    type = c("correlation", "covariance", "partial"),
    plot = TRUE, na.action, demean = TRUE, ...)
	
	

training.z.scaled	= gLogR[1:floor(nN*2/3)]
validation.z.scaled	= gLogR[1+floor(nN*2/3):(nN-1)]

plot(full.z.scaled,type="o")
nLags = 40
acf(full.z.scaled)

# gRDiff = diff(full.z.scaled,1)
# plot(gRDiff,type="o") # plot of first differences
# acf(gRDiff,xlim=c(1,24)) # plot of first differences, for 24 lags


# # assess data
# sink(strLog)
# summary(r)
# sink()
# file.show(strLog)

# fit = fitdistr(gLogR,"normal")

# # plot auto-correlation
# acf(y)

# hist(r,main="distribhist()ution of r",xlab="r")
# # plot(logy)
# plot.ts(gLogR)

# # fit AR model
# model = ar(logy, method = "ols")
# yPredict = predict(model, n.ahead = 25)
# z = ts.union(logy, yPredict$pred)

# plot(yPredict$pred)
# # export tot file 
# cat("yPredict", yPredict$pred, file = strFileOut, sep = "\n", append = FALSE)


# full.z.scaled = (full.z - mean(full.z))/ sd(full.z)



# AND <- c(rep(0,7),1)
# OR <- c(0,rep(1,7))
# binary.data <- data.frame(expand.grid(c(0,1), c(0,1), c(0,1)), AND, OR)
# print(net <- neuralnet(AND+OR~Var1+Var2+Var3,  binary.data, hidden=0, rep=10, err.fct="ce", linear.output=FALSE))


# for(i in 2010:2015){
  # print(i)
# }