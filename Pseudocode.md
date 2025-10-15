## --- Part 1: Offline Dimensionality Reduction ---

# This part is done once before training begins.
# Input: High-dimensional raw LLM features E_LLM for all items.

# 1. PCA Method (to create P-features)
#    - Compute the covariance matrix of E_LLM.
#    - Find the top 'd' eigenvectors (principal components) to form the matrix W_pca.
#    - Project the LLM features: E_P_all_items <-- E_LLM @ W_pca
#    - Save E_P_all_items for training.

# 2. Adapter Method (to create A-features)
#    - This is a trainable module, so no offline processing is needed.
#    - It will be initialized as a two-layer neural network within the main model.


## --- Part 2: Main Training Loop ---

# Initialization:
#   - Initialize all model parameters Theta (Adapter, LAFF, PGRC, SeqEncoder).
#   - Initialize M cluster prototypes {p_1, ..., p_M}.
#   - Load the pre-computed PCA features E_P_all_items.

FOR each training batch (sequences S, positive items P, negative items N):

    # --- Step 1: Feature Representation ---
    # Create two views of the input sequences.
    # A-View (Adapter):
    E_A_seq <-- Adapter(E_LLM_raw(S))
    # P-View (PCA):
    E_P_seq <-- E_P_all_items(S)

    # --- Step 2: LAFF Module (Denoising) ---
    # Apply the Low-Amplitude Frequency Filter to both views.
    FUNCTION LAFF(E):
        E_freq <-- FFT(E)
        // Selectively process low-amplitude components to reduce noise
        E_freq_filtered <-- Filter(E_freq)
        E_denoised <-- iFFT(E_freq_filtered)
        RETURN E_denoised
    END FUNCTION

    E_A_denoised <-- LAFF(E_A_seq)
    E_P_denoised <-- LAFF(E_P_seq)

    # --- Step 3: PGRC Module (Clustering and Loss Calculation on A-View) ---
    # A) Intra-cluster Attention
    // Assign each item feature in the sequence to one of K clusters
    Cluster_Assignments <-- ArgMax(E_A_seq @ W_centroids)
    Mask_Matrix <-- CreateMask(Cluster_Assignments)
    // Apply Multi-Head Attention within each cluster independently
    H_k <-- MultiHeadAttention(E_A_seq * Mask_k) for k in 1..K
    // Aggregate results to get enhanced features
    E_A_C <-- Sum(H_k * Mask_k for k in 1..K)

    # B) Prototype-guided Contrastive Clustering Loss (L_PGRC)
    // Create a concatenated view for contrastive learning
    E_con <-- Concat([E_A_C, E_A_seq])
    // Get a global representation for each sequence in the batch
    E_global <-- MeanPool(E_con, dimension=sequence_length)

    // b1. Clustering Loss (L_cluster)
    // Calculate probability distribution over M prototypes
    Similarities <-- (E_global @ p_prototypes.T) / nu
    Probabilities <-- Softmax(Similarities)
    L_cluster <-- -Mean(Sum(Probabilities * Log(Probabilities)))

    // b2. Contrastive Loss (L_contrast)
    // L2-normalize global features
    Z <-- L2_Normalize(E_global)
    // Calculate InfoNCE loss between augmented views (implicit in E_con)
    L_contrast <-- InfoNCE_Loss(Z)

    // b3. Combine for PGRC Loss
    L_PGRC <-- L_contrast + eta * L_cluster

    # --- Step 4: Sequence Encoding and Main Losses ---
    # Process the denoised sequences to get user representations
    User_Repr_A <-- SequenceEncoder(E_A_denoised)
    User_Repr_P <-- SequenceEncoder(E_P_denoised)

    # A) Contrastive Loss between dual views (L_CL)
    L_CL <-- InfoNCE_Loss(User_Repr_A, User_Repr_P)

    # B) BPR Loss for Recommendation (L_BPR)
    User_Repr_Unified <-- Combine(User_Repr_A, User_Repr_P)
    L_BPR <-- BPR_Loss(User_Repr_Unified, Positive_Items(P), Negative_Items(N))

    # --- Step 5: Total Loss and Optimization ---
    # Combine all losses
    L_total <-- L_BPR + lambda_1 * L_CL + lambda_2 * L_PGRC

    # A) Update model parameters
    Update Theta using gradient descent on L_total

    # B) Update Cluster Prototypes
    // Dynamically update prototypes based on current feature distributions
    Seq_Global_Repr <-- MeanPool(E_A_seq, dimension=sequence_length)
    Prototype_Assignments <-- ArgMax(Seq_Global_Repr @ p_prototypes.T)
    FOR m in 1..M:
        // Update each prototype to be the mean of features assigned to it
        p_m <-- Mean(Seq_Global_Repr[Prototype_Assignments == m])
    END FOR

END FOR

 