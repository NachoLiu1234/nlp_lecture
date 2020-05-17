import jieba


chinese = """5月7日，泰晶转债开盘暴跌30%而暂停交易。
收盘前的最后三分钟交易重新打开（按照规定，尾盘三分钟，可转债不设涨跌幅限制），泰晶转债最终跌幅扩大至47.68%。
此次暴跌源于5月6日晚间泰晶科技发布的“关于提前赎回泰晶转债的提示性公告”，再结合此前再升转债宣布赎回时的暴跌，反衬出当前可转债市场的非理性狂热。
可转债就是可以选择转股的债券。
一般期限为6年左右，可转股日为发行后6个月，具体要看各家发行的报告。
发行后半年投资者可以选择转股，按照票面金额来转股，例如100元的票面，转股价格20元，当前股价为40元，则对应转股价值为200元。
若转债价格高于200元，就会产生正的转股溢价率。
转债的纯债价值可以对投资者起到保护作用，一旦股价下跌过多，没有了转股价值，投资者至少可以以纯债的形式持有。""".split('\n')
english = """On May 7, the opening of Taijing Convertible Bonds plunged 30% and trading was suspended. 
The trading was reopened in the last three minutes before the close (according to the regulations, there is no limit for the rise and fall of convertible bonds for three minutes at the end of the day), and the final decline of Taijing Convertible Bonds expanded to 47.68%. 
This plunge originated from the "Informative Announcement on Redemption of Taijing Convertible Bonds in Advance" released by Taijing Technology on the evening of May 6, combined with the plunge when the previous redemption of convertible bonds was announced to redeem, reflecting the current convertible bonds Irrational fanaticism in the market.
Convertible bonds are bonds that can be converted into shares. 
The general term is about 6 years, and the conversion date is 6 months after the issue, depending on the report issued by each company. 
Investors can choose to convert shares half a year after the issuance and convert the shares according to the face value. 
For example, the face value of 100 yuan, the conversion price of 20 yuan, the current stock price of 40 yuan, then the corresponding value of 200 yuan. 
If the convertible bond price is higher than 200 yuan, a positive conversion premium rate will be generated. 
The pure debt value of convertible bonds can play a protective role for investors. Once the stock price falls too much, there is no conversion value, and investors can at least hold it in the form of pure debt.""".split('\n')
chinese = [['<start>'] + jieba.lcut(el) + ['<end>'] for el in chinese]
english = [['<start>'] + [e for e in jieba.lcut(el) if e != ' '] + ['<end>'] for el in english]
