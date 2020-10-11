require(phytools)
require(proxy)
require(Quartet)
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

#for (m in modes) {
#  d <- 0
#  dist_ <- NULL
#  for (i in 0:7) {
#    embeddings <- read.delim(paste('embeddings/language_embeddings_0_',i,'_',m,'.tsv',sep=''),sep='\t',row.names=1,header=F)
#    if (i == 0) {
#      dist_ <- dist(embeddings,'cosine')
#    }
#    else {
#      dist_ <- dist_ + dist(embeddings,'cosine')
#    }
#    tree <- nj(dist_)
#    quartets <- PairSharedQuartetStatus(tree,keep.tip(ref.tree,tree$tip.label))
#    d <- d + 1-(quartets[3]/(quartets[2]-quartets[5]))
#  }
#  d <- d/8
#  print(m)
#  print(d)
#}

tree <- read.newick('LDN_tree.newick')
quartets <- PairSharedQuartetStatus(tree,keep.tip(ref.tree,tree$tip.label))
print('LDN')
print(1-(quartets[3]/(quartets[2]-quartets[5])))

ref.tree <- keep.tip(ref.tree,tree$tip.label)
#ref.tree$edge.length <- rep(.1,length(ref.tree$edge))

#boxlabel<-function(x,y,text,cex=.5,bg="transparent",offset=0){
#  w<-strwidth(text)*cex*1.1
#  h<-strheight(text)*cex*1.4
#  os<-offset*strwidth("W")*cex
#  rect(x+os,y-0.5*h,x+w+os,y+0.5*h,col=bg,border=0)
#  text(x,y,text,cex=cex,pos=4,offset=offset,font=1)
#}
#plot(ref.tree,'c',cex=.001)

#pp<-get("last_plot.phylo",envir=.PlotPhyloEnv)
#N<-Ntip(ref.tree)
#par(fg="black")
#for(i in 1:Ntip(ref.tree)) {
#  rgb=mds[ref.tree$tip.label[i],]
#  boxlabel(pp$xx[i],pp$yy[i],ref.tree$tip.label[i],bg=rgb(rgb[1],rgb[2],rgb[3])) 
#}

tree <- nj(dists[["langPOSsemetym"]])
tree$edge.length <- NULL

tikz('LPSE_tree',width=3,height=5)

par(fg="transparent")
plotTree(tree)#,ylim=c(0,length(tree$tip.label)))
lastPP<-get("last_plot.phylo",env=.PlotPhyloEnv)

par(fg="black")

tt<-gsub("_"," ",tree$tip.label)
text(lastPP$xx[1:length(tt)],lastPP$yy[1:length(tt)],
     tt,cex=0.6,col='black',pos=4,offset=0.1,font=1)

dev.off()


mds <- cmdscale(dists[[m]],k=3)
for (i in 1:ncol(mds)) {
  mds[,i]<-(mds[,i]-min(mds[,i]))/(max(mds[,i])-min(mds[,i]))
  #mds[,i] <- 1/(1+exp(-mds[,i]))
}

lang.key <- read.csv('../lang_key.csv',header=F,sep='\t')

#mds$lang <- rownames(mds)
#merged <- merge(mds,lang.key,by.x='row.names',by.y='V5')
#merged <- merged[,c(1,2,3,4,9,10)]
#merged <- unique(merged)
#row.names(merged) <- merged[,1]
#merged <- merged[,2:6]

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