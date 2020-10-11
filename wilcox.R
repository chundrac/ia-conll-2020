collated <- read.csv('collated_output.tsv',sep='\t',header=F)

collated.long <- data.frame(error=c(collated[,5],collated[,7],collated[,9],collated[,11],collated[,13]),model=c(rep('L',nrow(collated)),rep('LP',nrow(collated)),rep('LPS',nrow(collated)),rep('LPSE',nrow(collated)),rep('B',nrow(collated))))

aggregate(error ~ model, collated.long, FUN=mean)

pairwise.wilcox.test(collated.long$error,collated.long$model,paired=T,p.adj='bonf')