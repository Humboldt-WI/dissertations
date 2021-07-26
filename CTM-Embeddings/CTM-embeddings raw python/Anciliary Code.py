import numpy as np
import pandas as pd
import random
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import random
from dash.exceptions import PreventUpdate
import re
import requests as request


K = 25

def read(path):
    data = np.matrix(pd.read_csv(path, header=None))
    return data


def clean_string(s):
    s = s.translate(str.maketrans('', '', '[]{}\'\"@.,:;?!=-+/\\&`*#^'))
    return s


def scale(x):
    for i in range(len(x)):
        x[i] = (x[i]-np.mean(x[i]))/(np.full( (1,len(x[i])), np.var(x[i])**0.5)[0])
    return x


def make_proper(data, length):
    c = np.zeros((len(data), length))
    for i in range(len(data)):
        a = data[i, 0].split()
        b = [float(n) for n in a]
        c[i] = b
    return c


def ATM_emb():
    meta_raw = np.matrix(pd.read_csv('ATM_meta.txt', delimiter='\t', header=None))
    meta_NEW_2 = []
    au_dir = {}
    au_dir_i = 0
    for i in meta_raw:
        cat_list = i[0, 0].replace(' {', '%[')
        full = cat_list.rsplit('%', 1)
        cats = {}
        cat_list = full[1].translate(str.maketrans('', '', '[]{}\'\":,'))
        cat_list = cat_list.split()
        ind = 0
        for i in range(int(len(cat_list) / 2)):
            cats[cat_list[ind]] = float(cat_list[ind + 1])
            ind += 2
        name_nr = str(full[0]).rsplit(' ', 1)
        name = name_nr[0]
        au_dir[name] = au_dir_i
        nr = int(name_nr[1])
        meta_NEW_2.append((name, nr, cats))
        au_dir_i += 1

    au_emb_raw = np.array(pd.read_csv('ATM_embeddings.txt', delimiter='\t', header=None))
    au_emb = []
    for i in range(int(len(au_emb_raw))):
        a = str(au_emb_raw[i]).translate(str.maketrans('', '', "[]()\'"))
        a = a.split(' ')
        a = np.array(a).astype(float)
        au_emb.append(a)

    return meta_NEW_2, au_emb


def CTM_emb():
    # scaled Lambda
    # Theta
    # author embeddings
    # Betas as topic-word distributions
    # word_inv list

    word_inv_raw = str(pd.read_csv('words_inv.txt', delimiter='\t'))
    word_inv_raw = word_inv_raw.split()
    words_inv = []
    ind = 3
    for i in range(int(len(word_inv_raw) / 2 - 2)):
        words_inv.append((int(clean_string(word_inv_raw[ind])), clean_string(word_inv_raw[ind + 1])))
        ind += 2
    print(words_inv)
    meta_raw = np.matrix(pd.read_csv('authors.txt', delimiter='\t', header=None))
    meta = []
    au_dir = {}
    au_dir_i = 0
    for i in meta_raw:
        cat_list = i[0, 0].replace(' {', '%[')
        full = cat_list.rsplit('%', 1)
        cats = {}
        cat_list = full[1].translate(str.maketrans('', '', '[]{}\'\":,'))
        cat_list = cat_list.split()
        ind = 0
        for i in range(int(len(cat_list) / 2)):
            cats[cat_list[ind]] = float(cat_list[ind + 1])
            ind += 2
        name_nr = str(full[0]).rsplit(' ', 1)
        name = name_nr[0]
        au_dir[name.replace(' ','')] = au_dir_i
        nr = int(name_nr[1])
        meta.append((name, nr, cats))
        au_dir_i += 1
    cum = np.zeros(len(meta)+1)
    tot = 0
    for i in range(len(cum)-1):
        tot += int(meta[i][1])
        cum[i+1] = tot
    cum = cum.astype(int)
    lam = read('lam_raw.txt')
    lam = make_proper(lam, K)
    theta = lam.copy()
    lam = np.transpose(lam)
    lam = scale(lam)
    lam = np.transpose(lam)

    for i in range(len(theta)):
        theta[i] = np.exp(theta[i])/sum(np.exp(theta[i]))
    auth_embedding = []
    hom = []
    for j in range(len(cum)-1):
        div = np.full((1, K), (cum[j + 1] - cum[j]))[0]
        au_ma = theta[cum[j]:cum[j + 1]]

        aggregate = []
        homogeneity = []
        length = len(au_ma[:, 0])
        for k in range(K):
            so = np.mean((au_ma[:, k]))
            aggregate.append(so)
            homogeneity.append(np.std(au_ma[:, k]))
        div = np.full((1, K), sum(aggregate))[0]
        auth_embedding.append(np.array(aggregate)/div)

        hom.append(homogeneity)
    beta = read('beta_numbers.txt')
    beta = make_proper(beta, len(words_inv))

    topic_words = []
    for j in range(K):
        s = list(beta[j])
        s_order = []
        for i in range(15):
            rounded = round(max(s) * 1000) / 1000
            s_order.append((words_inv[s.index(max(s))][1]))
            s[s.index(max(s))] = -1000
        #s_order = np.concatenate(s_order)
        topic_words.append(s_order)

    return meta, theta, beta, words_inv, lam, auth_embedding, topic_words, au_dir, hom


meta_ctm, theta, beta, words_inv, lam, ctm, topic_words, au_dir, hom = CTM_emb()
meta_atm, atm = ATM_emb()

topics_atm = [[('graph', 0.026917765302258076), ('graphs', 0.024854199437089844), ('polynomials', 0.01364085889383161), ('polynomial', 0.012436260008556688), ('number', 0.011152184493483926), ('paper', 0.00869807335954402), ('degree', 0.008548859861996772), ('prove', 0.008535582972636447), ('vertex', 0.007898386850922867), ('result', 0.007142742627325094), ('vertices', 0.006923188748564399), ('order', 0.006730000744299254), ('problem', 0.006578921135704353), ('functions', 0.0062606397256497566), ('curves', 0.00561847435197247)],
[('cohomology', 0.03424839703571942), ('groups', 0.021076238554897934), ('varieties', 0.020031717920019707), ('projective', 0.019062593893231148), ('curve', 0.017037363509516967), ('spaces', 0.01237605046644925), ('p-adic', 0.01051088393837224), ('category', 0.010339573478964227), ('prove', 0.009907199706849222), ('de', 0.009769149692380455), ('give', 0.009544742021662217), ('variety', 0.008416293440641745), ('toric', 0.007920746151645244), ('dialogue', 0.007871645474573552), ('etale', 0.007802945059166217)],
[('systems', 0.008602091898961411), ('method', 0.00734056151265936), ('models', 0.006931947517149136), ('spider', 0.00661535948448363), ('function', 0.0063873339260013765), ('functions', 0.005264042416520897), ('results', 0.005050441153336566), ('system', 0.004839139769341751), ('two-body', 0.004777413568846213), ('difference', 0.004775889095938667), ('logics', 0.004757046877584203), ('case', 0.004634491571779287), ('wave', 0.004569437769728256), ('obtained', 0.004379384935289661), ('structures', 0.004333254849552638)],
[('clusters', 0.02823518680205423), ('cosmic', 0.020727089234083715), ('energy', 0.019215594152496768), ('cluster', 0.018525908577440083), ('rays', 0.013355709463634326), ('sources', 0.009792741935816847), ('omega', 0.009300190461114015), ('events', 0.008246941315557598), ('high', 0.008186285834900284), ('group', 0.008105382024548292), ('limits', 0.007429509277408303), ('groups', 0.007339735670399495), ('ray', 0.007329463696356714), ('gamma', 0.0071612784874703446), ('primitive', 0.0066562893237170845)],
[('stars', 0.02895570307588757), ('star', 0.012926819238266139), ('stellar', 0.011361878024513055), ('-', 0.00819137091766117), ('data', 0.008035438857739323), ('planets', 0.007898757587636406), ('planet', 0.0061769596553892196), ('lines', 0.005725700629040567), ('hd', 0.00563019848422455), ('dwarfs', 0.005239653551734377), ('results', 0.005123033108388986), ('binary', 0.005020404249164267), ('sample', 0.00479719711195336), ('mass', 0.004788760836826252), ('emission', 0.0046678405644504945)],
[('algorithm', 0.010158806093734818), ('model', 0.00894469426705352), ('data', 0.008508128841772706), ('paper', 0.008254545980433669), ('method', 0.008093669597411982), ('control', 0.0077507401938722836), ('approach', 0.007198724957381025), ('methods', 0.006639699582438891), ('models', 0.006431522574466201), ('algorithms', 0.006299518598958458), ('performance', 0.005698085937010215), ('network', 0.005477071883070981), ('based', 0.005220205276623002), ('networks', 0.00507705308906194), ('learning', 0.005033901870133361)],
[('solar', 0.02459522305124114), ('disk', 0.009591236766560781), ('model', 0.0078002077429142), ('coronal', 0.007787078520944417), ('disks', 0.006208068208026155), ('flux', 0.005932707928548644), ('data', 0.0058128038869933825), ('disease', 0.005765073438788722), ('protoplanetary', 0.00561540743808622), ('fast', 0.005176120001719363), ('region', 0.005020203280286625), ('images', 0.004993357034749879), ('open', 0.004953429850351063), ('methods', 0.00488160800904751), ('method', 0.00486136844872404)],
[('radio', 0.02434743988347047), ('x-ray', 0.02256813869006823), ('emission', 0.012860049721275885), ('bursts', 0.011853172177052826), ('source', 0.011748385355240105), ('sources', 0.010972944853659227), ('pulsars', 0.010770501057177102), ('observations', 0.010422779726925624), ('data', 0.008473814412664602), ('burst', 0.008208151094278828), ('pulsar', 0.007858631500044051), ('array', 0.007614550366314202), ('frb', 0.007376455990606105), ('mhz', 0.006735825921109535), ('frequency', 0.006618640607268792)],
[('magnetic', 0.027823526901183605), ('field', 0.020902425221360887), ('transition', 0.016870887443365994), ('phase', 0.014995960166601069), ('temperature', 0.00903800339515248), ('lattice', 0.008587725809061295), ('state', 0.008557855327235965), ('behavior', 0.008125125817340182), ('critical', 0.007666107210707905), ('josephson', 0.0069227202362744595), ('range', 0.006324047139086514), ('superconductors', 0.00606819648516439), ('superconducting', 0.005995222852692164), ('vortex', 0.0059614553693571595), ('frequency', 0.005767586532968784)],
[('network', 0.015961914726665246), ('fractional', 0.013897668609664444), ('social', 0.01160863223250222), ('networks', 0.00715362346384956), ('analysis', 0.0067838756313995), ('continuation', 0.0059213175002318695), ('brand', 0.0057946055628842795), ('semantic', 0.0054764692766291895), ('pricing', 0.0053160446908089775), ('online', 0.004884734679977284), ('importance', 0.004786756020110992), ('based', 0.004781336817114515), ('envelope', 0.004730064418994664), ('centrality', 0.004717516759109103), ('option', 0.0046966284426009156)],
[('mass', 0.009420467157995232), ('supernova', 0.009164360439386265), ('explosion', 0.008262072021342059), ('membrane', 0.007893373668176268), ('supernovae', 0.007841926683222747), ('models', 0.007672637893404693), ('ejecta', 0.0075358264261329595), ('collapse', 0.0072777894411990025), ('protein', 0.0070165392980166014), ('neutron', 0.006444499648619587), ('white', 0.006031306072827349), ('star', 0.006029951930985179), ('core', 0.005961608279439267), ('simulations', 0.005701032010512537), ('observed', 0.005669452475702237)],
[('galaxies', 0.01631902111635705), ('galaxy', 0.010098966074639129), ('sn', 0.008755410063268307), ('-', 0.008577937725711639), ('mass', 0.007202536973354877), ('observations', 0.007178854356388156), ('emission', 0.006809353433443321), ('star', 0.006376990834557747), ('type', 0.0059260733924462695), ('optical', 0.005880797425744501), ('find', 0.0057804893125768435), ('spectra', 0.005280266100070827), ('stellar', 0.00511535662294156), ('redshift', 0.004970994481983461), ('ia', 0.004896424877041654)],
[('theory', 0.017282379308754686), ('model', 0.016375813052261785), ('solutions', 0.011846977232910906), ('equations', 0.00908798098308249), ('field', 0.009069413405134882), ('equation', 0.007848377725860233), ('models', 0.007727005077880064), ('general', 0.007624878500455186), ('study', 0.007384185247620273), ('theories', 0.00705955315673265), ('space', 0.006716258960906744), ('case', 0.006651763986191244), ('terms', 0.006359087660883761), ('quantum', 0.006125775636262947), ('limit', 0.0055256370390349505)],
[('pi', 0.010971791883396708), ('collisions', 0.01037138998805461), ('states', 0.010187954950030265), ('chiral', 0.007991000375941653), ('interaction', 0.00796128385091702), ('nuclear', 0.007653022479201449), ('data', 0.007029626591568743), ('meson', 0.006662855155279503), ('state', 0.0065011712019338985), ('resonance', 0.006405147257599282), ('reaction', 0.006401004489439126), ('find', 0.0063146403685690414), ('heavy', 0.006204984230464837), ('decay', 0.006098051045898094), ('experimental', 0.0059815062394389966)],
[('states', 0.016230316597905452), ('quantum', 0.014071855898951081), ('phase', 0.01396965743262585), ('spin', 0.013390999069906399), ('symmetry', 0.011413262932337976), ('state', 0.010332791734510806), ('topological', 0.009717189420743916), ('phases', 0.008073423999916545), ('magnetic', 0.007359148638974835), ('band', 0.006880433118875921), ('energy', 0.00671067420911757), ('interactions', 0.006624713464217474), ('interaction', 0.006555334714049843), ('graphene', 0.006458355319350656), ('hall', 0.006196467927738449)],
[('gravitational', 0.011573405046869288), ('ligo', 0.01014716052452425), ('advanced', 0.008737653516434048), ('quantum', 0.007387314674852656), ('wave', 0.007314833790582421), ('noise', 0.006712234856978012), ('search', 0.006472273013670875), ('detectors', 0.006285032214279133), ('gravitational-wave', 0.006063909422541422), ('waves', 0.005996041929738789), ('bps', 0.005505541880760517), ('binary', 0.0054434836392832624), ('sensitivity', 0.0052720843080652155), ('observing', 0.0052628520857405745), ('detection', 0.005249307022509934)],
[('solid', 0.016336099957585223), ('algebras', 0.01513122368546109), ('sphere', 0.009951559222719411), ('pure', 0.00794038438734304), ('monotone', 0.007723253487907199), ('noncommutative', 0.0073781804547367715), ('bd', 0.007217514765076565), ('systems', 0.007096701891493523), ('integrals', 0.007072190329295734), ('liouville', 0.0069527745767205915), ('galilean', 0.006683480447035999), ('momentum', 0.00608720832963296), ('integral', 0.0056553707371958814), ('integrable', 0.005387567097402383), ('instability', 0.005361627128355224)],
[('neutrino', 0.014849818372632922), ('dark', 0.014565441410067027), ('matter', 0.012351099984626357), ('model', 0.011953245048978186), ('mass', 0.010198537348240127), ('cosmological', 0.007716827256107242), ('energy', 0.0073007988777721924), ('scale', 0.007282930981439528), ('data', 0.006517688815070836), ('standard', 0.006306034144232358), ('spectrum', 0.005992473908015049), ('measurements', 0.005622486976416088), ('background', 0.005450998989575454), ('large', 0.005281765364534711), ('neutrinos', 0.005268267512753158)],
[('estimates', 0.017218711336680452), ('li', 0.00995643715731854), ('singularities', 0.008072946775464652), ('orthogonal', 0.007579798514061946), ('reactions', 0.007545171491395861), ('hypersurfaces', 0.0071141704909752496), ('transfer', 0.0069110328764548454), ('range', 0.005936106713138618), ('hera', 0.005805865450602945), ('functions', 0.005627131454242552), ('accumulation', 0.005556511439680785), ('weighted', 0.005494987428654168), ('cross-sections', 0.005485789374321409), ('obtain', 0.005385100318536857), ('wrt', 0.005248206277280469)],
[('system', 0.017720566862072788), ('dynamics', 0.01597701880464459), ('three-body', 0.01312743880035231), ('systems', 0.01151030750360423), ('stochastic', 0.01092180571242395), ('gas', 0.010539937642633883), ('results', 0.010301603265805129), ('quantum', 0.008860035726179842), ('interactions', 0.007979850698147538), ('scattering', 0.00785344676755621), ('parameters', 0.007451696822315718), ('probability', 0.007180818404138221), ('regime', 0.006990542536040542), ('study', 0.006472999252467711), ('classical', 0.006415942921385591)],
[('detector', 0.02052962276436656), ('data', 0.016110378684736984), ('-', 0.01126609220305011), ('decays', 0.010104589649313645), ('measurements', 0.009584123423120941), ('decay', 0.008403627531122062), ('experiment', 0.00808632137228277), ('resolution', 0.0076976266929537395), ('pm', 0.007551067651355134), ('lhcb', 0.006670274786207887), ('measurement', 0.006536601160159215), ('time', 0.0061878748554633324), ('measured', 0.006159365951248237), ('system', 0.0056061651274643964), ('energy', 0.0054886127938898075)],
[('magnetic', 0.024128842687208746), ('field', 0.016998560740299387), ('simulations', 0.010850846679585266), ('accretion', 0.010689346759124007), ('gas', 0.01019023040302734), ('flow', 0.00975321112598313), ('disk', 0.008878964238963382), ('large', 0.008828544943761556), ('density', 0.007693008317888466), ('plasma', 0.0075592733725802766), ('hall', 0.007533762884131175), ('bilayer', 0.00709842690094283), ('model', 0.0069660269038755215), ('molecular', 0.006560798811641058), ('turbulence', 0.006090234340541753)],
[('black', 0.03918612457903383), ('hole', 0.02488860626704141), ('holes', 0.01625900089780286), ('field', 0.0161725095648634), ('energy', 0.01534038163708876), ('magnetic', 0.012621161412151801), ('mass', 0.011174105209960141), ('fields', 0.010539288361198459), ('cme', 0.008241287500985735), ('massive', 0.008171975610740253), ('gravitational', 0.007854085241545915), ('gravity', 0.00717196193708561), ('particles', 0.007100562908928894), ('tau', 0.005855461587811541), ('corona', 0.005725861910098842)],
[('time', 0.017971713049777533), ('entropy', 0.011656571763564976), ('finite', 0.007314352165743925), ('entanglement', 0.0072415560684786565), ('random', 0.006756825206902677), ('simple', 0.006580754975102543), ('model', 0.005422473163473427), ('terms', 0.005201617362426342), ('sequence', 0.004892078343192732), ('results', 0.004597016629379385), ('theory', 0.004495648624223556), ('present', 0.004302486330414779), ('series', 0.0042575112546789605), ('local', 0.004100860914251018), ('general', 0.0037791709885981115)],
[('quantum', 0.01675920030930548), ('field', 0.01658763904679444), ('light', 0.014308623177037818), ('laser', 0.01349821848181303), ('energy', 0.013442349145134277), ('optical', 0.009664871073679152), ('photon', 0.009197107241324233), ('polarization', 0.009043628851791302), ('scattering', 0.008744104128093356), ('control', 0.007896543801504494), ('atomic', 0.007327152251500056), ('photons', 0.007110770627111839), ('coherent', 0.006842393821508134), ('electron', 0.005966078688606557), ('large', 0.005958918916515672)]
]


def resemble(emb, num):
    betaC = beta.copy()
    embc = list(emb).copy()
    auth_words = []
    for i in range(num):
        m = np.max(embc)
        m_ind = embc.index(m)
        embc[m_ind] = 0
        w = []
        BC = list(betaC[m_ind])
        for j in range(15):
            maximum = np.max(BC)
            max_ind = BC.index(maximum)
            w.append(words_inv[max_ind][1])
            BC[max_ind] = 0
        wr = random.sample(w, 5)
        auth_words.append(wr)
    #print(auth_words)
    return auth_words


def resemble_atm(x, nr):
    emb = list(x)
    auth_words = []
    for i in range(nr):
        T = []
        m_ind = emb.index(np.max(emb))
        emb[m_ind] = 0
        for e in range(len(topics_atm[m_ind])):
            T.append(topics_atm[m_ind][e][0])
        auth_words.append(random.sample(T, 5))
    return auth_words


def mass(emb):
    emb = list(emb)
    mass = 0
    for i in range(len(emb)):
        mass += np.max(emb)
        emb[emb.index(np.max(emb))] = 0
        if mass > 0.5:
            return i+1


def entropy(emb):
    s = 0
    for i in range(len(emb)):
        if emb[i] > 0:
            s += -emb[i]*np.log(emb[i])
        else:
            s += 0
    return s
# ATM entropy: 1.4130466823098162
# ATM ent.var: 0.27280902341488183

# CTM entropy: 1.6494478999179074
# CTM ent.var: 0.2746876079034711
E = 0
V = []
for j in range(len(atm)):
    #print(j)
    e = entropy(ctm[j])
    E += e
    V.append(e)
print("HERE: ", E/len(atm))
print("VAR: ", np.var(V))

A = -1
B = -1
C = -1
D = -1
true_intruder = -1
true_target = -1
true_bestmatch = -1
true_second = -1
solutions = []


def snippit(au, meta_x):
    query_author = '%22' + au + '%22'
    start = int(np.floor(random.uniform(0, 1) * meta_x[au_dir[au.replace(' ','')]][1]))
    c = []
    url = 'http://export.arxiv.org/api/query?search_query=' + query_author + '&start=' + str(start) + '&max_results=1'
    data = request.get(url)
    data = str(data.content)
    data = data.replace('<title>', 'XSX')
    data = data.replace('</summary>', 'XSX')
    data = data.replace('<entry>', 'DSFGC')
    data_split = data.split('DSFGC')
    for i in range(len(data_split) - 1):
        data_split_part = data_split[i + 1].split('XSX')
        title_abstract = str(data_split_part[1])
        title_abstract = title_abstract.replace('</title>', '')
        title_abstract = title_abstract.replace('<summary>', '_title_split_')
        title_abstract = title_abstract.replace('\\n', ' ')
    title_abstract = title_abstract.split('_title_split_')
    c.append(clean_string(title_abstract[0]))
    c.append(clean_string(title_abstract[1]))

    return c


def intrude(x, emb, which_meta, resemb):
    target = emb[x]
    m_target = mass(target)
    L = 1
    U = 5
    if m_target in [1,2]:
        L = 1
        U = 2
    elif m_target == 3:
        L= 2
        U = 3
    elif m_target == 4:
        L = 3
        U = 4
    dist_ctm = []
    for a in emb:
        dist_ctm.append((1 / (2 ** 0.5)) * (sum(((target ** 0.5 - a ** 0.5) ** 2))) ** 0.5)
    dist_ctm_sorted = []
    for i in range(len(emb)):
        ctm_min = np.min(dist_ctm)
        dist_ctm_sorted.append(which_meta[dist_ctm.index(ctm_min)][0])
        dist_ctm[dist_ctm.index(ctm_min)] = 2
    reps = 0
    while True:
        r_one_rand = int(1+np.floor(10*random.uniform(0, 1)))
        r_two_rand = int(r_one_rand+np.floor(10*random.uniform(0, 1)))
        recom_one = list(emb[au_dir[(dist_ctm_sorted[r_one_rand]).replace(' ', '')]])
        recom_two = list(emb[au_dir[(dist_ctm_sorted[r_two_rand]).replace(' ', '')]])
        rand = int(40 + np.floor(random.uniform(0, 1) * 10))
        intruder = dist_ctm_sorted[rand]
        meta_index = au_dir[intruder.replace(' ', '')]
        emb_intruder = list(emb[meta_index])
        if (mass(recom_one) in [L,U] and
            mass(recom_two) in [L,U] and
            mass(emb_intruder) in [L,U]):
            break
        if reps > 100:
            print('RECALL')
            intrude(int(np.floor(random.uniform(0, 1)*238)), emb, which_meta, resemb)
            break

        reps += 1

    global A, B, C, D
    A = resemb(target, mass(target))
    B = resemb(recom_one, mass(recom_one))
    C = resemb(recom_two, mass(recom_two))
    D = resemb(emb_intruder, mass(emb_intruder))
    solutions.append({'intrusion':D, 'target':A, 'best_recom':B, 'best_recom_two':C})
    group = [A,B,C,D]
    group_shuffled = random.sample(group, 4)
    #for i in range(len(group_shuffled)):
        #print(group_shuffled[i])
    abc = ['A', 'B', 'C', 'D']
    global true_intruder, true_target, true_bestmatch, true_second
    true_intruder = abc[group_shuffled.index(D)]
    true_target = abc[group_shuffled.index(A)]
    true_bestmatch = abc[group_shuffled.index(B)]
    true_second = abc[group_shuffled.index(C)]
    s = snippit(which_meta[x][0], which_meta)
    return group_shuffled, s


which_one = -1
def CTMvsATM():
    global which_one
    which_one = random.uniform(0,1)
    if which_one > 0.5:
        # ATM if > 0.5
        ra = int(np.floor(random.uniform(0, 1)*238))
        shuffle, s = intrude(ra, atm, meta_atm, resemble_atm)
        np.savetxt('ATMorCTM.txt', [which_one], fmt='%s')
    else:
        # CTM if < 0.5
        ra = int(np.floor(random.uniform(0, 1)*238))
        shuffle, s = intrude(ra, ctm, meta_ctm, resemble)
    a, b, c, d = shuffle

    return a, b, c, d, s

def compare(a, b, depth):
    first_recom = -len(a)
    for j in range(len(a)):
        fitsr_recom_i = -depth
        target = j
        dist_ctm=[]
        dist_atm=[]
        for ctm, atm in zip(a, b):
            dist_ctm.append((1/(2**0.5))*(sum(((a[target]**0.5-ctm**0.5)**2)))**0.5)
            dist_atm.append((1/(2**0.5))*(sum(((b[target]**0.5-atm**0.5)**2)))**0.5)
        dist_ctm_sorted=[]
        dist_atm_sorted=[]
        for i in range(depth):
            ctm_min=np.min(dist_ctm)
            atm_min=np.min(dist_atm)
            dist_ctm_sorted.append(meta_ctm[dist_ctm.index(ctm_min)][0])
            dist_atm_sorted.append(meta_atm[dist_atm.index(atm_min)][0])
            #if dist_ctm_sorted[i] == dist_atm_sorted[i]:
            #    first_recom+=1
            dist_ctm[dist_ctm.index(ctm_min)]=2
            dist_atm[dist_atm.index(atm_min)]=2
        #print(dist_ctm_sorted)
        #print(dist_atm_sorted)
        for e in range(depth):
            if dist_ctm_sorted[e] in dist_atm_sorted:
                first_recom+=1
                fitsr_recom_i+=1
        #if fitsr_recom_i == -depth+1:
            #print(j, meta_ctm[j][0])
            #print(dist_ctm_sorted)
            #print(dist_atm_sorted)

    #print(first_recom/((depth-1)*len(a)))
    return first_recom/((depth-1)*len(a))


#print(compare(ctm, atm, 119))

# 69 Katherine E. Whitaker
#1-   0.1932
#5-   0.297
#10-  0.3525
#25 - 0.458
#119- 0.7227246830935764
#238- 1.0


def cleanup(a):
    a = a.replace("], [",'SPLITTER_XXX')
    a = a.translate(str.maketrans('', '', "[]\'"))
    a = a.replace(',', ' ')
    a = a.split('SPLITTER_XXX')
    a_new = []
    for i in range(len(a)):
        a_new.append(a[i])
    a_new = random.sample(a_new, len(a_new))
    a_new_with_breaks = []
    for j in range(len(a_new)):
        a_new_with_breaks.append(a_new[j])
        a_new_with_breaks.append(html.Br())
    return a_new_with_breaks

app = dash.Dash(__name__)
app.title = 'Intrusion Metric Survey'

app.layout = html.Div([
    html.Div([
        html.H4(id='abstractTitle'),
        html.P(id='abstractText', style={'margin-bottom':'2%'}),
        html.Div('A',id='canvasA', style={'background-color':'rgba(193, 220, 255, 1)','padding':'10px'}),
        html.Div('B', id='canvasB', style={'padding':'10px','background-color':'rgba(236, 242, 251, 1)'}),
        html.Div('C', id='canvasC', style={'background-color':'rgba(193, 220, 255, 1)','padding':'10px'}),
        html.Div('D', id='canvasD', style={'padding':'10px','background-color':'rgba(236, 242, 251, 1)'  }),
    ], id='canvas', style={'text-align':'center'}),
    html.Div([
    html.Button('A', id="A", style={'width':'15%'}),
    html.Button('B', id="B", style={'width':'15%'}),
    html.Button('C', id="C", style={'width':'15%'}),
    html.Button('D', id="D", style={'width':'15%'})],
        id='buttons', style={'text-align':'center','margin-top':'1%'}),
    html.H2(id='answerText', style={'text-align':'center'}),
    html.Button('Generate Task', id='init', style={'position':'relative', 'color':'rgb(0, 109, 255)'}),

    ], style={'background-color':'rgba(145, 148, 179, 0.2)'})

app.scripts.append_script({
    'external_url': '/assets/style.css'
})

@app.callback(
    Output(component_id='canvasA', component_property='children'),
    Output(component_id='canvasB', component_property='children'),
    Output(component_id='canvasC', component_property='children'),
    Output(component_id='canvasD', component_property='children'),
    Output(component_id='abstractTitle', component_property='children'),
    Output(component_id='abstractText', component_property='children'),
    Input(component_id='init', component_property='n_clicks')
)
def paint(n_clicks):
    a, b, c, d, s = CTMvsATM()
    a = cleanup(str(a))
    b = cleanup(str(b))
    c = cleanup(str(c))
    d = cleanup(str(d))
    return a, b, c, d, s[0], s[1]

@app.callback(
    Output('answerText', 'children'),
    Output('A', 'n_clicks'),
    Output('B', 'n_clicks'),
    Output('C', 'n_clicks'),
    Output('D', 'n_clicks'),
    Input('A', 'n_clicks'),
    Input('B', 'n_clicks'),
    Input('C', 'n_clicks'),
    Input('D', 'n_clicks'),
    Input('init', 'n_clicks')
)
def buttonAnswer(A, B, C, D, init):
    print(A, B, C, D, init)
    ctx = dash.callback_context
    button_id = ''
    if ctx.triggered:
        if not ctx.triggered[0]['prop_id'].split('.')[0] == 'init':
            if which_one > 0.5:
                wo = 'ATM'
            else:
                wo = 'CTM'
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == true_intruder:
                button_id = 'Model:'+wo+'; Correctly identified intruder'
            else:
                abc = ['Target Author', 'Best Match', 'Second Best Match']
                set_of_answers = [true_target, true_bestmatch, true_second]
                button_id = 'Model: '+wo+'; False Answer, intruder is '+true_intruder+'. You selected the '+abc[set_of_answers.index(button_id)]
        else:
            button_id = ''
    return button_id, 0, 0, 0, 0

#if __name__ == '__main__':
#   app.run_server(debug=True)


