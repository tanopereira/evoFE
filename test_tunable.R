source("R/make_tunable.R")
# mock evo_evaluators environment just to pass the initial check
evo_evaluators <- new.env()
evo_evaluators$xgboost <- list(train_func=function(...) list())

make_tunable("xgboost",list(eta=list(type="numeric",lower=.1,upper=1),nrounds=list(type="integer",value=500,tunable=F),max_depth=list(type="integer",lower=0,upper=20),grow_policy=list(type="discrete",values=c("depthwise","lossguide")),subsample=list(type="numeric",lower=.1,upper=1),colsample_bytree=list(type="numeric",lower=.1,upper=1),max_delta_step=list(type="numeric",lower=1,upper=10),min_child_weight=list(type="integer",lower=1,upper=50)))
message("Success!")
