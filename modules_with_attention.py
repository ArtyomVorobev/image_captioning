from beheaded_inception import beheaded_inception_v3,BeheadedInception3

enc_model = beheaded_inception_v3().train(False) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):

  def __init__(self,emb_dim,enc_model=enc_model,out_size=8,enc_dim=2048,device=device):

      super(Encoder, self).__init__()
      self.encoder = enc_model
      self.emb_dim = emb_dim
      self.out_size = out_size
      self.linear = nn.Linear(enc_dim,emb_dim)
      self.transforms = transforms.Compose([
            transforms.Resize((299,299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
      self.device = device

  def CustomTransform(self,batch,img2ind=False):

      images = torch.tensor([])
      
      if img2ind == True: 
          batch = [index_to_img[ind.item()] for ind in batch]

      
      for img in batch:

          image = Image.open(img).convert('RGB')
          image = self.transforms(image)
          images = torch.cat((images, image.unsqueeze(0)))

      return images 


  def forward(self,input,transforms=True):
      # out = (batch_size, 2048, out_size, out_size)

      if transforms == True:
          input = self.CustomTransform(input,img2ind=True)   #    (batch_size, 3, image_size, image_size)
      
      else: 
          input = [input]
          input = self.CustomTransform(input,img2ind=False)   #    (1, 3, image_size, image_size)

      input = input.to(self.device)

      
      output = self.encoder(input) 
                                 
      #getting output for attention
      (output_for_attn, image_vector, _ ) = output
      
      output_for_attn = output_for_attn.permute(0, 2, 3, 1)                      #(batch_size, out_size, out_size, 2048)

      output_for_attn = torch.flatten(output_for_attn, start_dim=1, end_dim=2)   #(batch_size, pixels_num, 2048)

     
                      
      image_vector = self.linear(image_vector)                #(batch_size,emb_dim)
      image_vector = image_vector.unsqueeze(0)                #(1, batch_size, emb_dim)

      return output_for_attn, image_vector 



class Attention(nn.Module):

    def __init__(self,dec_hid_dim,att_size=8,enc_out_dim=2048,Temperature=10):
        super(Attention, self).__init__()

        self.att_size = att_size
        self.dec_hid_dim = dec_hid_dim
        self.temperature = Temperature
        self.dec_att = nn.Linear(dec_hid_dim, att_size)
        self.enc_att = nn.Linear(enc_out_dim, att_size)    
        self.att = nn.Linear(att_size, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self,decoder_out,encoder_out):
        # dec_out (num_layers * num_directions, batch, hidden_size)
        # enc_out (batch_size, 64=pixels_num, 2048)

        
        att1 = self.dec_att(decoder_out.permute(1,0,2))                     # (batch_size, num_layers * num_directions, att_size)

         
        att1 = att1.mean(dim=1).unsqueeze(1)                                 # (batch_size, 1, att_size)       
                                                      
        
        att2 = self.enc_att(encoder_out)                   # (batch_size, pixels_num, att_size)
                                
        att = att1 + att2                                  # (batch_size, pixels_num ,att_size)
        att = self.att(self.tanh(self.relu(att)))          # (batch_size, pixels_num, 1)
        att = att.squeeze(2)                               # (batch_size, pixels_num)

        weights = self.softmax(att/self.temperature)             # (batch_size, pixels_num) 

        attention = encoder_out * weights.unsqueeze(2)           # (batch_size, pixels_num, 2048)
        attention = attention.mean(dim=2)                        # (batch_size, pixels_num)

        return attention 


class DecoderWithAttention(nn.Module):

    def __init__(self, output_dim, emb_dim, att_size, encoder_out, dec_hid_dim, dropout, attention,num_layers=2):
        
        super(DecoderWithAttention, self).__init__()
        
        self.embedding = nn.Embedding(output_dim,emb_dim)
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.encoder_out = encoder_out
        self.dec_hid_dim = dec_hid_dim
        self.att_size = att_size
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers 
        self.attention = attention 
        self.rnn = nn.LSTM(2*emb_dim + att_size,
             dec_hid_dim,num_layers=num_layers,
            dropout=dropout)

        self.out = nn.Linear(dec_hid_dim,output_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input, hidden,cell_state, encoder_outputs,image_vector):
      
        input = input.unsqueeze(0)                                          # because only one word, no words sequence 
        embedded = self.dropout(self.embedding(input))                      # (1,batch_size,emb_dim)    
        embedded = torch.cat((embedded,image_vector),dim=2)                 # (1,batch_size,emb_dim)

        # get weighted sum of encoder_outputs             
        attention = self.attention(hidden,encoder_outputs).unsqueeze(0)     # (1,batch_size,att_size)  (att_size = pixels_num)
        # concatenate weighted sum and embedded, break through the RNN
        rnn_input = torch.cat((embedded,attention),dim=2)                   # (1,batch_size, att_size + 2*emb_dim)
   

        
        rnn_out, (next_hidden, cell_state) = self.rnn(rnn_input,(hidden,cell_state))                   # (batch_size, dec_hid_dim)  

        out = self.out(rnn_out)                                             # (1,batch_size, output_dim)  
        

        return out, (next_hidden, cell_state)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device,dec_hid_dim,vocab=word_to_index,last_enc_dim=2048,pixels_num=8*8,max_len=25):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_len = max_len
        self.vocab = vocab
        self.num_layers = decoder.num_layers
        self.enc_transform = nn.Linear(last_enc_dim, dec_hid_dim)
        self.pixels_transform = nn.Linear(pixels_num, decoder.num_layers)
        self.softmax = nn.Softmax(dim=2)
        

    def forward(self, img, text,teacher_forcing_ratio = 0.5,generate_caption=False):          
        
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]


        if generate_caption == True: 
            # only for 1 image
            max_len = self.max_len
            # tensor to store predicted words 
            words = torch.zeros(max_len)
            #tensor to store decoder outputs
            outputs = torch.tensor([])

            #first input to the decoder is the <sos> tokens
            input = torch.tensor([self.vocab['#START#']]).to(self.device)
      
            transform = False 
          


        else:
            # Again, now batch is the first dimention instead of zero
            batch_size = len(img)                        
            max_len = text.shape[0]                           # (max_len,batch_size)
            trg_vocab_size = self.decoder.output_dim

            # tensor to store predicted words 
            words = torch.zeros(max_len, batch_size).to(self.device)
            #tensor to store decoder outputs
            outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
            
            #first input to the decoder is the <sos> tokens
            input = text[0,:].to(self.device)
            transform = True 
            

      
        #output of the encoder is used as the initial hidden state of the decoder
        enc_states,image_vector = self.encoder(img,transform)
        
        
        hidden = enc_states                                      # (batch_size, pixels_num, enc_last_dim)

        hidden = self.enc_transform(hidden)                      # (batch_size, pixels_num,dec_hid_dim) 

        hidden = hidden.permute(0,2,1)                           # (batch_size, dec_hid_dim, pixels_num)
        
        hidden = self.pixels_transform(hidden)                   # (batch_size, dec_hid_dim,num_layers)

        hidden = hidden.permute(2,0,1).contiguous()              # (num_layers, batch_size,dec_hid_dim)
         
        cell_state = hidden                              
        
        words[0] = self.vocab['#START#']
        for t in range(1,max_len):    
          
            output,(hidden,cell_state) = self.decoder(input, hidden,cell_state, enc_states,image_vector)   

            teacher_force = random.random() < teacher_forcing_ratio
            preds = self.softmax(output)
            top1 = preds.max(2)[1]
            
            if generate_caption == True:
                words[t] = int(top1[0].item())
                input = top1[0]
                

            else:
                words[t] = top1[0]
                
                outputs[t] = output[0]
                # top1 or ground truth
                input = (text[t] if teacher_force else top1[0])

        return outputs,words

