# ================================
# 0. Load libraries
# ================================
library(reticulate)
py_require("sentence-transformers")
library(readr)
library(dplyr)
library(igraph)
library(proxy)
library(stringr)
library(DT)

# Load embedding model
st <- import("sentence_transformers")$SentenceTransformer
model <- st("all-MiniLM-L6-v2")


# ================================
# 1. Load Dataset
# ================================
dataset <- read_csv("climate-fever.csv")
View(dataset)

claims <- dataset %>%
  select(claim_id, claim, claim_label) %>%
  distinct(claim_id, .keep_all = TRUE) %>%
  filter(claim_label %in% c("SUPPORTS", "REFUTES"))
View(claims)

# Split dataset
claims_support <- claims %>% filter(claim_label == "SUPPORTS")
claims_refute  <- claims %>% filter(claim_label == "REFUTES")


# ================================
# 2. Generate Embeddings
# ================================
emb_s <- py_to_r(model$encode(claims_support$claim))
emb_r <- py_to_r(model$encode(claims_refute$claim))
View(emb_s)
View(emb_r)

# ================================
# 3. Compute Similarity Matrices
# ================================
sim_s <- as.matrix(proxy::simil(emb_s, method="cosine"))
sim_r <- as.matrix(proxy::simil(emb_r, method="cosine"))
View(sim_s)
View(sim_r)

# ================================
# 4. Convert similarity to edges
# ================================
to_edges <- function(sim_matrix, threshold) {
  df <- as.data.frame(as.table(sim_matrix))
  colnames(df) <- c("from", "to", "weight")
  
  df %>%
    filter(from != to & weight > threshold) %>%
    mutate(pair = paste0(pmin(from,to), "_", pmax(from,to))) %>%
    distinct(pair, .keep_all = TRUE) %>%
    select(-pair)
}

edges_s <- to_edges(sim_s, 0.6)
edges_r <- to_edges(sim_r, 0.6)


# Replace numeric index with real claim_id
edges_s$from <- claims_support$claim_id[as.numeric(edges_s$from)]
edges_s$to   <- claims_support$claim_id[as.numeric(edges_s$to)]

edges_r$from <- claims_refute$claim_id[as.numeric(edges_r$from)]
edges_r$to   <- claims_refute$claim_id[as.numeric(edges_r$to)]


# ================================
# 5. Create Graphs
# ================================
g_s <- graph_from_data_frame(edges_s, directed = FALSE)
g_r <- graph_from_data_frame(edges_r, directed = FALSE)

V(g_s)$label <- claims_support$claim[match(V(g_s)$name, claims_support$claim_id)]
V(g_r)$label <- claims_refute$claim[match(V(g_r)$name, claims_refute$claim_id)]


# ================================
# 6. Visualize ORIGINAL graphs
# ================================
par(mfrow=c(1,2))

plot(g_s, vertex.size=5, vertex.label=NA,
     main="Supported Claims Network (Original)")

plot(g_r, vertex.size=5, vertex.label=NA,
     main="Refuted Claims Network (Original)")


# ================================
# 7. Louvain Community Detection
# ================================
comm_s <- cluster_louvain(g_s)
comm_r <- cluster_louvain(g_r)


# ================================
# 8. Visualize Communities
# ================================
par(mfrow=c(1,2))

plot(comm_s, g_s, vertex.size=5, vertex.label=NA,
     main="Supported Claims (Communities)")

plot(comm_r, g_r, vertex.size=5, vertex.label=NA,
     main="Refuted Claims (Communities)")


# ================================
# 9. Centrality Metrics
# ================================
get_metrics <- function(g) {
  data.frame(
    claim_id = V(g)$name,
    degree = degree(g),
    betweenness = betweenness(g),
    claim_text = V(g)$label
  )
}

metrics_s <- get_metrics(g_s)
metrics_r <- get_metrics(g_r)


# ================================
# 10. MOST CENTRAL CLAIM IN EACH COMMUNITY
# ================================
get_community_topics <- function(graph, comm_object, class_name) {
  
  communities <- membership(comm_object)
  results <- list()
  
  for (c in unique(communities)) {
    nodes_in_comm <- names(communities[communities == c])
    
    df <- data.frame(
      claim_id = nodes_in_comm,
      text = V(graph)$label[match(nodes_in_comm, V(graph)$name)],
      degree = degree(graph)[match(nodes_in_comm, V(graph)$name)],
      betweenness = betweenness(graph)[match(nodes_in_comm, V(graph)$name)]
    )
    
    # Most central node = highest degree
    top_node <- df[which.max(df$degree), ]
    
    results[[paste("Community", c)]] <- list(
      size = nrow(df),
      central_claim = top_node,
      all_claims = df
    )
  }
  
  return(results)
}

topics_support <- get_community_topics(g_s, comm_s, "SUPPORT")
topics_refute  <- get_community_topics(g_r, comm_r, "REFUTE")
topics_support
topics_refute

# ================================
# 11. Print Topics for Each Community
# ================================
cat("\n============================\n")
cat("SUPPORTED CLAIM COMMUNITIES\n")
cat("============================\n")

for (name in names(topics_support)) {
  cat("\n", name, " (", topics_support[[name]]$size, "claims )\n")
  print(topics_support[[name]]$central_claim)
}

cat("\n============================\n")
cat("REFUTED CLAIM COMMUNITIES\n")
cat("============================\n")

for (name in names(topics_refute)) {
  cat("\n", name, " (", topics_refute[[name]]$size, "claims )\n")
  print(topics_refute[[name]]$central_claim)
}


# ================================
# 12. Cross-Network Contradictions
# ================================
cross_sim <- proxy::simil(emb_s, emb_r, method="cosine")
cross_df <- as.data.frame(as.table(as.matrix(cross_sim)))
colnames(cross_df) <- c("support_id","refute_id","similarity")

strong_pairs <- cross_df %>%
  filter(similarity >= 0.90) %>%
  mutate(
    support_claim_id = claims_support$claim_id[support_id],
    refute_claim_id = claims_refute$claim_id[refute_id],
    support_text = claims_support$claim[support_id],
    refute_text = claims_refute$claim[refute_id]
  )

View(strong_pairs)
write.csv(strong_pairs, "contradicting_claim_pairs.csv", row.names=FALSE)
