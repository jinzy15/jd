from transformer.Models.HRED import *

class LHRED(HRED):

    def train(self,session,len_session,criterion,encoder_optimizer,context_optimizer,decoder_optimizer):
        if(len_session>1):
            in_session = session[0:-1]
            encoder_hidden = self.encoder.initHidden()
            context_hidden = self.context.initHidden()
            in_session,session_hidden = self.encoder(in_session,encoder_hidden)
            islast = False
            loss = 0
            for idx,sentence in enumerate(torch.transpose(session_hidden,1,0)):
                if(idx == len_session-2):
                    islast = True
                if(idx<len_session-1):
                    context_input = sentence  #这里的context_input就可以代表一句话的全部信息
                    context_output,context_hidden=self.context(context_input,context_hidden)
                    if(islast):
                        decoder_hidden = context_hidden
                        decoder_input = torch.tensor([[SOS_token]], device=device)

                        target_sentence = session[len_session-1]
                        target_length = len(target_sentence)
                        use_teacher_forcing = True if random.random() < 0.5 else False
                        loss = 0
                        encoder_optimizer.zero_grad()
                        decoder_optimizer.zero_grad()
                        context_optimizer.zero_grad()

                        if use_teacher_forcing:
                            # Teacher forcing: Feed the target as the next input
                            for di in range(target_length):
                                if(int(target_sentence[di])==0):
                                    break
                                decoder_output,decode_hidden = self.decoder(decoder_input,decoder_hidden)

                                loss += criterion(decoder_output, Variable(torch.tensor([target_sentence[di]],device=device)))
                                # print(loss)


                                decoder_input = target_sentence[di]  # Teacher forcing
                                # print(decoder_input)
                                if int(decoder_input) == EOS_token:
                                    break
                        else:
                            # Without teacher forcing: use its own predictions as the next input
                            for di in range(target_length):
                                if(int(target_sentence[di])==0):
                                    break
                                decoder_output,decode_hidden = self.decoder(decoder_input,decoder_hidden)
                                topv, topi = decoder_output.data.topk(1)
                                ni = topi[0][0]
                                # print(decoder_output)
                                # print(ni)
                                decoder_input = torch.tensor(ni,device=device)
                                # print(decoder_output, Variable(torch.tensor([target_sentence[di]], device=device)))
                                encoder_optimizer.zero_grad()
                                decoder_optimizer.zero_grad()
                                context_optimizer.zero_grad()
                                loss += criterion(decoder_output, Variable(torch.tensor([target_sentence[di]], device=device)))
                                if int(ni) == EOS_token:
                                    break
                        loss.backward()
                        encoder_optimizer.step()
                        context_optimizer.step()
                        decoder_optimizer.step()
            return loss
        else:
            print('fuck')
            return 0