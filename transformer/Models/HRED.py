from transformer.Models.Models import *

class HRED(Models):
    def __init__(self, dataset,train_loader, test_loader, encoder,decoder,context):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.encoder = encoder
        self.decoder = decoder
        self.dataset = dataset
        self.context = context

    def train(self,session,len_session,criterion,encoder_optimizer,context_optimizer,decoder_optimizer):
        if(len_session>1):
            in_session = session[0:-1]
            tar_session = session[1:]
            encoder_hidden = self.encoder.initHidden()
            context_hidden = self.context.initHidden()
            # print (device)
            # print (type(in_session),type(encoder_hidden))
            in_session,session_hidden = self.encoder(in_session,encoder_hidden)

            for idx,sentence in enumerate(torch.transpose(session_hidden,1,0)):
                if(idx<len_session-1):
                    encoder_optimizer.zero_grad()
                    decoder_optimizer.zero_grad()
                    context_optimizer.zero_grad()

                    context_input = sentence[-1]  #这里的context_input就可以代表一句话的全部信息

                    context_output,context_hidden=self.context(context_input,context_hidden)
                    decoder_hidden = context_hidden
                    decoder_input = torch.tensor([[SOS_token]], device=device)
                    use_teacher_forcing = True if random.random() < 0.5 else False
                    target_sentence = tar_session[idx]
                    target_length = len(target_sentence)
                    loss = 0
                    if use_teacher_forcing:
                        # Teacher forcing: Feed the target as the next input
                        for di in range(target_length):
                            if(target_sentence[di]==0):
                                break
                            decoder_output,decode_hidden = self.decoder(decoder_input,decoder_hidden)

                            print(decoder_output, Variable(torch.tensor([target_sentence[di]],device=device)))
                            if (di >= self.dataset.lang.n_words) or (di < 0):
                                import pdb; pdb.set_trace()
                            loss = criterion(decoder_output, Variable(torch.tensor([target_sentence[di]],device=device)))
                            print(loss)
                            decoder_input = target_sentence[di]  # Teacher forcing
                            loss.backward(retain_graph = True)
                            encoder_optimizer.step()
                            context_optimizer.step()
                            decoder_optimizer.step()

                    else:
                        # Without teacher forcing: use its own predictions as the next input
                        for di in range(target_length):
                            if(target_sentence[di]==0):
                                break
                            decoder_output,decode_hidden = self.decoder(decoder_input,decoder_hidden)
                            topv, topi = decoder_output.data.topk(1)
                            ni = int(topi[0][0])
                            # print(decoder_output)
                            # print(ni)
                            decoder_input = torch.tensor(ni,device=device)

                            if (di >= self.dataset.lang.n_words) or (di < 0):
                                import pdb; pdb.set_trace()
                            print(decoder_output, Variable(torch.tensor([target_sentence[di]], device=device)))
                            loss += criterion(decoder_output, Variable(torch.tensor([target_sentence[di]], device=device)))
                            loss.backward(retain_graph=True)
                            encoder_optimizer.step()
                            context_optimizer.step()
                            decoder_optimizer.step()
                            if ni == EOS_token:
                                break

            return loss.item()
        else:
            print('fuck')
            return 0

    def trainEpoch(self,epoch_times = 1, print_every=10000, plot_every=100,save_every=1000,learning_rate=0.0001):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
        context_optimizer = optim.SGD(self.context.parameters(), lr=learning_rate)

        criterion = nn.CrossEntropyLoss()
        for epoch in range(epoch_times):
            n_iters = len(self.dataset)
            for iter,(session,len_session) in enumerate(self.dataset):
                loss = self.train(session,len_session,criterion,encoder_optimizer,context_optimizer,decoder_optimizer)
                print_loss_total += loss
                plot_loss_total += loss
                if (iter+epoch*n_iters)%print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print('%s epoch(%d / %d) iters(%d / %d) %d%% loss is %.4f' % (timeSince(start, ((iter+1)+epoch*n_iters)/ (epoch_times*n_iters)),epoch,epoch_times,
                                                 iter,n_iters ,((iter+1)+epoch*n_iters)/ (epoch_times*n_iters) * 100, print_loss_avg))

                if (iter+epoch*n_iters)% plot_every == 0:
                    plot_loss_avg = plot_loss_total / plot_every
                    plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0
                if (iter + epoch * n_iters) % save_every == 0:
                    torch.save(self.encoder.state_dict(), 'train_fruit/' + description + '_encoder.pkl')
                    torch.save(self.decoder.state_dict(), 'train_fruit/' + description + '_decoder.pkl')
                    torch.save(self.context.state_dict(), 'train_fruit/' + description + '_context.pkl')

    def trainIters(self,n_iters, print_every=100, evaluate_every = 100,save_every=100, learning_rate=0.0001):

        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
        context_optimizer = optim.SGD(self.context.parameters(), lr=learning_rate)

        criterion = nn.CrossEntropyLoss()
        training_sessions = [random.choice(self.dataset) for i in range(n_iters)]

        for iter in range(1, n_iters + 1):
            session,len_session = training_sessions[iter - 1]
            loss = self.train(session, len_session, criterion, encoder_optimizer, context_optimizer, decoder_optimizer)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))



            if iter % save_every == 0:
                torch.save(self.encoder.state_dict(), 'train_fruit/' + description + '_encoder.pkl')
                torch.save(self.decoder.state_dict(), 'train_fruit/' + description + '_decoder.pkl')
                torch.save(self.context.state_dict(), 'train_fruit/' + description + '_context.pkl')
                # plot_loss_avg = plot_loss_total / plot_every
                # plot_losses.append(plot_loss_avg)
                # plot_loss_total = 0

            if iter % evaluate_every == 0:
                session


    def evaluate(self,sentences,max):
        encoder_hidden = torch.zeros(1, batch_size, hidden_size, device=device)
        in_session,num_sentences = self.dataset.tensorsFromSession(sentences)
        in_session, session_hidden = self.encoder(in_session, encoder_hidden)
        max_length = max
        decoded_words = []
        context_hidden = self.context.initHidden()

        for idx, sentence in enumerate(torch.transpose(session_hidden, 1, 0)):
            context_input = sentence[-1]
            context_output, context_hidden = self.context(context_input, context_hidden)
        decoder_input = torch.LongTensor([[SOS_token]], device=device)  # SOS

        if use_cuda:
            decoder_input = decoder_input.cuda()

        decoder_hidden = context_hidden
        # decoder_output = decoder_input

        for di in range(max_length):
            # print(decoder_output,context_hidden)
            decoder_output , decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = int(topi[0][0])
            # print(ni)
            if ni == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(self.dataset.lang.index2word[ni])
            decoder_input = torch.LongTensor([[ni]], device=device).cuda()
        return decoded_words

    def sen2torch(self,origin_sess,torch_sess):
        pass


    def evaluateRandomly(self,n):
        for i in range(n):
            session = random.choice(self.dataset.sessions)
            print('>', session[0:-1])
            print('=', session[-1])
            output_words = self.evaluate(session[0:-1],MAX_LENGTH)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')