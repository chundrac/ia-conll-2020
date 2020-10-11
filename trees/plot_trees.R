require(phytools)
require(proxy)
require(tikzDevice)

ref.tree <- read.newick('glottolog_tips.newick')

modes <- c('lang',
           'langPOS',
           'langPOSsem',
           'langPOSsemetym')

cosdist <- function(x,y) {
  return(1 - (sum(x*y)/sqrt(sum(x^2)*sum(y^2)) ))
}

dists <- list()
for (m in modes) {
  dist_ <- NULL
  for (i in 0:7) {
    embeddings <- read.delim(paste('embeddings/language_embeddings_0_',i,'_',m,'.tsv',sep=''),sep='\t',row.names=1,header=F)
    if (i == 0) {
      dist_ <- dist(embeddings,'cosine')
    }
    else {
      dist_ <- dist_ + dist(embeddings,'cosine')
    }
  }
  dists[[m]] <- dist_/8
}

for (m in modes) {
  plot(nj(dists[[m]]))
}

for (m in modes) {
  tree <- nj(dists[[m]])
  quartets <- PairSharedQuartetStatus(tree,keep.tip(ref.tree,tree$tip.label))
  print(m)
  print(1-(quartets[3]/(quartets[2]-quartets[5])))
}

mds <- cmdscale(dists[[m]],k=3)
for (i in 1:ncol(mds)) {
  mds[,i]<-(mds[,i]-min(mds[,i]))/(max(mds[,i])-min(mds[,i]))
  #mds[,i] <- 1/(1+exp(-mds[,i]))
}

LDN.tree <- read.newick('LDN_tree.newick')

tikz('LDN_tree',width=3,height=5)

plot(LDN.tree)

dev.off()

tree <- ref.tree

colors <- c()
for (t in tree$tip.label) {
  vals <- mds[t,]
  colors <- c(colors,rgb(1-vals[1],vals[2],vals[3]))
}


tikz('IA_tree',width=3,height=5)
par(fg="transparent")
plotTree(tree)#,ylim=c(0,length(tree$tip.label)))
lastPP<-get("last_plot.phylo",env=.PlotPhyloEnv)

par(fg="black")

tt<-gsub("_"," ",tree$tip.label)
text(lastPP$xx[1:length(tt)],lastPP$yy[1:length(tt)],
     tt,cex=0.6,col=colors,pos=4,offset=0.1,font=1)

dev.off()