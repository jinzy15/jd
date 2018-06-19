class HRED_QA(object):
    def __init__(self,
                 groups=None,  # qa data in groups
                 dictionary=None,  # should be a HRED model preprocessed dictionary
                 id2word=None,
                 word2id=None,
                 encoder_file=None,
                 decoder_file=None,
                 context_file=None,
                 teacher_forcing_ratio=0.5,
                 hidden_size=512,
                 beam=1,
                 max_sentence_length=30,
                 context_layers=1,
                 attention_layers=1,
                 decoder_layers=1,
                 learning_rate=0.0001
                 ):
        self.groups = groups
        self.dictionary = dictionary
        self.word2id = word2id
        self.id2word = id2word
        self.encoder_file = encoder_file
        self.decoder_file = decoder_file
        self.context_file = context_file
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.hidden_size = hidden_size
        self.beam = beam
        self.max_sentence_length = max_sentence_length
        self.context_layers = context_layers
        self.attention_layers = attention_layers
        self.decoder_layers = decoder_layers
        self.learning_rate = learning_rate
        self.encoder_model = None
        self.decoder_model = None
        self.context_model = None

        # load word2id if not present but dictionary is string
        if self.dictionary and type(self.dictionary) == str and not self.word2id:
            dt = pkl.load(open(self.dictionary, 'r'))
            self.word2id = {d[0]: d[1] for d in dt}
            self.id2word = {d[1]: d[0] for d in dt}
            self.EOS_token = self.word2id['</s>']
            self.SOS_token = self.word2id['</d>']
            self.dictionary = dt

        self.create_or_load_models()

    # cerate models if they are none
    # load models if they are string
    def create_or_load_models(self):
        encoder_model = Module.EncoderRNN(len(self.word2id.keys()), self.hidden_size)
        decoder_model = AttnDecoderRNN(self.hidden_size,
                                       len(self.word2id.keys()), self.attention_layers, dropout_p=0.1)
        context_model = ContextRNN(self.hidden_size, len(self.word2id.keys()))

        if self.encoder_file and type(self.encoder_file) == str and os.path.exists(self.encoder_file):
            encoder_model.load_state_dict(torch.load(self.encoder_file))

        if self.decoder_file and type(self.decoder_file) == str and os.path.exists(self.decoder_file):
            decoder_model.load_state_dict(torch.load(self.decoder_file))

        if self.context_file and type(self.context_file) == str and os.path.exists(self.context_file):
            context_model.load_state_dict(torch.load(self.context_file))

        if use_cuda:
            encoder_model = encoder_model.cuda()
            decoder_model = decoder_model.cuda()
            context_model = context_model.cuda()

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.context_model = context_model

    # for hred, train should take the context of the previous turn
    # should return current loss as well as context representation

    def train(self, input_variable, target_variable,
              encoder, decoder, context, context_hidden,
              encoder_optimizer, decoder_optimizer, criterion,
              last, max_length=None):

        max_length = self.max_sentence_length
        encoder_hidden = encoder.initHidden()

        # encoder_optimizer.zero_grad() # pytorch accumulates gradients, so zero grad clears them up.
        # decoder_optimizer.zero_grad()

        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[self.SOS_token]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        # calculate context
        context_output, context_hidden = context(encoder_output, context_hidden)

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_output, encoder_outputs, context_hidden)
                loss += criterion(decoder_output[0], target_variable[di])
                decoder_input = target_variable[di]  # Teacher forcing

        else:

            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_output, encoder_outputs, context_hidden)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input

                # only calculate loss if its the last turn
                if last:
                    loss += criterion(decoder_output[0], target_variable[di])
                if ni == self.EOS_token:
                    break

        if last:
            loss.backward()

        # encoder_optimizer.step()
        # decoder_optimizer.step()

        if last:
            return loss.data[0] / target_length, context_hidden
        else:
            return context_hidden

    def indexesFromSentence(self, word2id, sentence):
        return [word2id.get(word, word2id['<unk>']) for word in sentence.split(' ') if len(word) > 0]

    def variableFromSentence(self, sentence=None, indexes=None):
        indexes = self.indexesFromSentence(self.word2id, sentence)
        indexes.append(self.EOS_token)
        # print len(indexes)
        result = Variable(torch.LongTensor(indexes).view(-1, 1), requires_grad=False)
        if use_cuda:
            return result.cuda()
        else:
            return result

    def variablesFromPair(self, pair):
        input_variable = self.variableFromSentence(indexes=pair[0])
        target_variable = self.variableFromSentence(indexes=pair[1])
        return (input_variable, target_variable)

    # return variables from group
    def variablesFromGroup(self, group):
        variables = [self.variableFromSentence(sentence=p) for p in group]
        return variables

    # training should proceed over each set of dialogs
    # which should be in variable groups = [u1,u2,u3...un]
    def trainIters(self, encoder=None, decoder=None,
                   context=None, print_every=500, plot_every=100, evaluate_every=500,
                   learning_rate=None):

        encoder = self.encoder_model
        decoder = self.decoder_model
        context = self.context_model
        learning_rate = self.learning_rate
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
        context_optimizer = optim.Adam(context.parameters(), lr=learning_rate)

        # TODO: experiment with decayed learning rate when the api is available
        # enc_scheduler = StepLR(encoder_optimizer,step_size=3000,gamma=0.7)
        # dec_scheduler = StepLR(decoder_optimizer,step_size=3000,gamma=0.7)
        # con_scheduler = StepLR(context_optimizer,step_size=3000,gamma=0.7)
        criterion = nn.NLLLoss()

        print
        "training started"
        iter = 0
        while True:
            iter += 1
            # training_pair = training_pairs[iter - 1]
            training_group = self.variablesFromGroup(random.choice(self.groups))
            # print len(training_group)
            context_hidden = context.initHidden()
            context_optimizer.zero_grad()
            encoder_optimizer.zero_grad()  # pytorch accumulates gradients, so zero grad clears them up.
            decoder_optimizer.zero_grad()
            for i in range(0, len(training_group) - 1):
                input_variable = training_group[i]
                target_variable = training_group[i + 1]
                last = False
                if i + 1 == len(training_group) - 1:
                    last = True

                if last:
                    loss, context_hidden = self.train(input_variable, target_variable, encoder,
                                                      decoder, context, context_hidden, encoder_optimizer,
                                                      decoder_optimizer, criterion, last)
                    print_loss_total += loss
                    plot_loss_total += loss
                    encoder_optimizer.step()
                    decoder_optimizer.step()
                    context_optimizer.step()
                else:
                    context_hidden = self.train(input_variable, target_variable, encoder,
                                                decoder, context, context_hidden, encoder_optimizer, decoder_optimizer,
                                                criterion, last)

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('steps %d loss %.4f' % (iter, print_loss_avg))

            if iter % (print_every * 3) == 0:
                # save models
                print
                "saving models"
                torch.save(encoder.state_dict(), self.encoder_file)
                torch.save(decoder.state_dict(), self.decoder_file)
                torch.save(context.state_dict(), self.context_file)

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

            if iter % evaluate_every == 0:
                self.evaluateRandomly(encoder, decoder, context)

                # enc_scheduler.step()
                # dec_scheduler.step()
                # con_scheduler.step()
                # showPlot(plot_losses)

    def evaluate(self, encoder, decoder, context, sentences, max_length=None,
                 beam=1):
        max_length = self.max_sentence_length
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        context_hidden = context.initHidden()

        for i, sentence in enumerate(sentences):
            last = False
            if i + 1 == len(sentences):
                last = True
            input_variable = self.variableFromSentence(sentence=sentence)
            input_length = input_variable.size()[0]
            encoder_hidden = encoder.initHidden()

            encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
            encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                         encoder_hidden)
                encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

            decoder_input = Variable(torch.LongTensor([[self.SOS_token]]))  # SOS
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            decoder_hidden = encoder_hidden

            # calculate context
            context_output, context_hidden = context(encoder_output, context_hidden)

            def decode_with_beam(decoder_inputs, decoder_hiddens, beam):
                new_decoder_inputs = []
                new_decoder_hiddens = []
                decoder_outputs = torch.FloatTensor().cuda() if use_cuda else torch.FloatTensor()
                # decoder_outputs_h = torch.FloatTensor()
                for i, decoder_input in enumerate(decoder_inputs):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hiddens[i], encoder_output, encoder_outputs, context_hidden)
                    # print decoder_output.data
                    # print decoder_outputs
                    decoder_outputs = torch.cat((decoder_outputs, decoder_output.data), 1)
                    # decoder_outputs_h = torch.cat((decoder_outputs_h,decoder_output[0]),1)
                    new_decoder_hiddens.append(decoder_hidden)

                topv, topi = decoder_outputs.topk(beam)
                nis = list(topi[0])
                nh = []  # decoder_hidden
                for ni in nis:
                    nip = ni % len(self.word2id.keys())  # get the word id
                    # if nip == EOS_token:
                    #    continue # or break?
                    decoder_input = Variable(torch.LongTensor([[nip]]))
                    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
                    new_decoder_inputs.append(decoder_input)
                    nh.append(new_decoder_hiddens[int((ni / len(self.word2id.keys())))])

                return new_decoder_inputs, nh, (nis[0] % len(self.word2id.keys()))

            decoder_inputs = [decoder_input]
            decoder_hiddens = [decoder_hidden]
            for di in range(max_length):
                decoder_inputs, decoder_hiddens, ni = decode_with_beam(decoder_inputs, decoder_hiddens, beam)
                if last:
                    if ni == self.EOS_token:
                        decoded_words.append('<eos>')
                        break
                    else:
                        decoded_words.append(self.id2word[ni])

        return decoded_words

    def evaluateRandomly(self, encoder, decoder, context, n=10):
        for i in range(n):
            group = random.choice(self.groups)
            for gr in group:
                print('>', gr)
            output_words = self.evaluate(encoder, decoder, context, group[:-1])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')

