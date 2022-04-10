'''
Author: your name
Date: 2022-04-10 00:11:08
LastEditTime: 2022-04-10 20:16:09
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /hw1/parse.py
'''
def readCommand (argv):
    from optparse import OptionParser
    usageStr = """
        USAGE: python train_model.py <options>
        Eamples:    (1) python train_model.py
                    (2) python train_model.py -h 256
                    (3) python train_model.py -l 0.01
    """
    parser = OptionParser(usageStr)
    parser.add_option('-i', '--input_layer', dest='input_layer', type='int',
                        help="Input Data Size", default=784)
    parser.add_option('-s', '--hidden_layer', dest='hidden_layer', type='int',
                        help="Hidden Layer Size", default=512)
    parser.add_option('-o', '--output_layer', dest='output_layer', type='int',
                        help="Output Layer Size", default=10)
    parser.add_option('-l', '--learningrate', dest='learningrate', type='float',
                        help="Learning rate", default=0.1)
    parser.add_option('-r', '--reg', dest='reg', type='float',
                        help="Regularization strength", default=1e-4)
    parser.add_option('-t', '--test_data', dest='test_data', type='int',
                        help="input test data", default=False)
    parser.add_option('-d', '--lr_decay', dest='lr_decay', type='float',
                        help="learning rate decay", default=0.95)
    parser.add_option('-e', '--epoch', dest='epoch', type='int',
                        help="number of epoch", default=50)
    (options, junkArgs) = parser.parse_args(argv)
    args = dict()
    args['input_layer'] = options.input_layer
    args['hidden_layer'] = options.hidden_layer
    args['output_layer'] = options.output_layer
    args['learningrate'] = options.learningrate
    args['reg'] = options.reg
    args['test_data'] = options.test_data
    args['lr_decay'] = options.lr_decay
    args['epoch'] = options.epoch
    return args
