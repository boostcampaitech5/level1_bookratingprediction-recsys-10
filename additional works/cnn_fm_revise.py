# 기존 유저/상품 벡터와 이미지 벡터를 결합하여 FM으로 학습하는 모델을 구현합니다.
class CNN_FM2(torch.nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = np.array([len(data['user2idx']), len(data['isbn2idx'])], dtype=np.uint32)
        self.embedding = FeaturesEmbedding(self.field_dims, args.cnn_embed_dim)
        self.img_feature = nn.Linear(512, 32, bias=False)
        self.fm = FactorizationMachine(
                                       input_dim=(args.cnn_embed_dim * 2) + (32), 
                                       latent_dim=args.cnn_latent_dim,
                                       )
    
    def forward(self, x):
        user_isbn_vector, img_vector = x[0], x[1]
        user_isbn_feature = self.embedding(user_isbn_vector)   
        img_feature = self.img_feature(img_vector)                 
        feature_vector = torch.cat([user_isbn_feature.view(-1, user_isbn_feature.size(1) * user_isbn_feature.size(2)), 
                                    img_feature
                                    ], dim=1)
        output = self.fm(feature_vector)
        return output.squeeze(1)