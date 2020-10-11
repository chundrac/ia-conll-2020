require(phytools)
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

#LDN tree

LDN.dist <- read.csv('LDN_dist.tsv',header=T,row.names=1,sep='\t')
LDN.dist <- as.dist(LDN.dist)
LDN.nj <- nj(LDN.dist)
plot(LDN.nj)
write.tree(LDN.nj,file='LDN_tree.newick')

#embedding trees

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
  tree <- nj(dists[[m]])
  plot(tree)
  write.tree(tree,file=paste(m,'.newick',sep=''))
}