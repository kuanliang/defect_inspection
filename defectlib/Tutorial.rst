
.. code:: ipython2

    from defectlib import extract_bnfeatures_from_defect, train_svm_classifier, plot_tsne
    import numpy as np
    
    from defectlib import load_tensors
    from defectlib import load_tensors_all
    from defectlib import combine_shuffle_tensors, display_tensor, keras_transform, make_model, train_model
    from defectlib import Config, remove_sn, remain_sn
    from matplotlib import image
    from IPython.display import Image
    from tqdm import tqdm, trange
    import matplotlib.pyplot as plt
    import defectlib
    import cv2
    import os
    import numpy as np
    
    from bokeh.plotting import figure, output_notebook
    output_notebook()
    from bokeh.plotting import figure, show
    
    %matplotlib inline


.. parsed-literal::

    Using TensorFlow backend.



.. raw:: html

    
        <div class="bk-root">
            <a href="http://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
            <span id="3828bcc4-ea9b-4eb4-bc80-38741a81377d">Loading BokehJS ...</span>
        </div>




pre-trained google Inception-v3 deep learning model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convolution Neural Network主要是兩個部分組成

1. 特徵萃取層 (Convolution layers, activation, max or average pooling)
2. 分類層 (fully connected layers)

Transfer Learning 的方法則是利用google inception
model訓練好的特徵萃取層來轉換原始圖片 (Inception-v3 model:
從2012年起利用ImageNet比賽的資料訓練進行訓練，1M照片/1000類別)

.. code:: ipython2

    from IPython.display import Image, display
    Image('images/07_inception_flowchart_bottleneck.png')




.. image:: output_2_0.png



Transfer Learning Flowchart
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython2

    from IPython.display import Image, display
    Image('images/defect_inception_nn.png')




.. image:: output_4_0.png



.. code:: ipython2

    # transfer leraning
    from tensorflow.python.platform import gfile
    def create_graph(model_path):
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            # parse read file with parseFro
            graph_def.ParseFromString(f.read())
            # Load teh Inception-V3 file
            _ = tf.import_graph_def(graph_def, name='')

.. code:: ipython2

    def extract_bottleneck_features(list_images):
        '''extract buttleneck features from a list of images
        
        Notes:
            
        
        Args:
            list_images (list): a list of path_to_images
        
        Return:
            features (numpy array): an 2 dimensional numpy array,
                                    each row represents a transformed feature of an image
            
        '''
        # set up the expected transformed feature number
        nb_features = 2048
        
        # initial feature numpy array
        features = np.empty((len(list_images),nb_features))
        
        labels = []
        
        # specified the inception model
        create_graph('./inception_dec_2015/tensorflow_inception_graph.pb')
        
        
        with tf.Session() as sess:
            
            # Get a reference to the pool_3 layer
            next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
            #return next_to_last_tensor
            for ind, image in enumerate(list_images):
                if (ind%100 == 0):
                    print('Processing %s...' % (image))
                if not gfile.Exists(image):
                    tf.logging.fatal('File does not exist %s', image)
    
                image_data = gfile.FastGFile(image, 'rb').read()
                print type(image_data)
                print image_data.shape
                predictions = sess.run(next_to_last_tensor,
                    {'DecodeJpeg/contents:0': image_data}
                )
                
                print predictions.shape
                
                features[ind,:] = np.squeeze(predictions)
                # labels.append(re.split('_\d+', image.split('/')[1])[0])
                # print labels
    
        return features

利用google 訓練好的 inception model 將原始瑕疵影像轉換到2048維度 (embedding)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython2

    features, labels, sns, images = extract_bnfeatures_from_defect('./ben1214/CL0401/', comb=True)


.. parsed-literal::

    there are 26 images inside CL0401_2A_c0
    Processing ./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_0_2A_0.jpg...
    there are 60 images inside CL0401_2A_c1
    Processing ./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y64830U17GY4TA_2A_0.jpg...
    there are 207 images inside CL0401_2A_c4
    Processing ./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6452037UGY4TA_2A_0.jpg...
    Processing ./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65013JQKGY4TA_2A_5.jpg...
    Processing ./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050P7DGY4RB_2A_3.jpg...
    there are 29 images inside CL0401_6A_c0
    Processing ./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_0_6A_0.jpg...
    there are 36 images inside CL0401_6A_c4
    Processing ./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y6452037UGY4TA_6A_0.jpg...
    there are 39 images inside CL0401_6A_c7
    Processing ./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y649617UMH3RPA_6A_0.jpg...
    there are 19 images inside CL0401_8A_c1
    Processing ./ben1214/CL0401/8A/CL0401_8A_c1/CL0401_Recombination_F3Y64830U17GY4TA_8A_0.jpg...
    there are 17 images inside CL0401_8A_c4
    Processing ./ben1214/CL0401/8A/CL0401_8A_c4/CL0401_Recombination_F3Y6452037UGY4TA_8A_0.jpg...
    there are 15 images inside CL0401_8A_c7
    Processing ./ben1214/CL0401/8A/CL0401_8A_c7/CL0401_Recombination_F3Y649617UMH3RPA_8A_0.jpg...


.. code:: ipython2

    features.shape




.. parsed-literal::

    (448, 2048)



.. code:: ipython2

    labels.shape




.. parsed-literal::

    (448,)



.. code:: ipython2

    plot_tsne(features, labels, images, perplexity=20, interactive=False)



.. image:: output_11_0.png


.. code:: ipython2

    plot_tsne(features, labels, sns, perplexity=20, interactive=True, images=images)



.. raw:: html

    
    
        <div class="bk-root">
            <div class="plotdiv" id="bbbf02e2-2946-41b4-8bb1-e1cd8277265a"></div>
        </div>
    <script type="text/javascript">
      
      (function(global) {
        function now() {
          return new Date();
        }
      
        var force = "";
      
        if (typeof (window._bokeh_onload_callbacks) === "undefined" || force !== "") {
          window._bokeh_onload_callbacks = [];
          window._bokeh_is_loading = undefined;
        }
      
      
        
        if (typeof (window._bokeh_timeout) === "undefined" || force !== "") {
          window._bokeh_timeout = Date.now() + 0;
          window._bokeh_failed_load = false;
        }
      
        var NB_LOAD_WARNING = {'data': {'text/html':
           "<div style='background-color: #fdd'>\n"+
           "<p>\n"+
           "BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \n"+
           "may be due to a slow or bad network connection. Possible fixes:\n"+
           "</p>\n"+
           "<ul>\n"+
           "<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\n"+
           "<li>use INLINE resources instead, as so:</li>\n"+
           "</ul>\n"+
           "<code>\n"+
           "from bokeh.resources import INLINE\n"+
           "output_notebook(resources=INLINE)\n"+
           "</code>\n"+
           "</div>"}};
      
        function display_loaded() {
          if (window.Bokeh !== undefined) {
            Bokeh.$("#bbbf02e2-2946-41b4-8bb1-e1cd8277265a").text("BokehJS successfully loaded.");
          } else if (Date.now() < window._bokeh_timeout) {
            setTimeout(display_loaded, 100)
          }
        }
      
        function run_callbacks() {
          window._bokeh_onload_callbacks.forEach(function(callback) { callback() });
          delete window._bokeh_onload_callbacks
          console.info("Bokeh: all callbacks have finished");
        }
      
        function load_libs(js_urls, callback) {
          window._bokeh_onload_callbacks.push(callback);
          if (window._bokeh_is_loading > 0) {
            console.log("Bokeh: BokehJS is being loaded, scheduling callback at", now());
            return null;
          }
          if (js_urls == null || js_urls.length === 0) {
            run_callbacks();
            return null;
          }
          console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
          window._bokeh_is_loading = js_urls.length;
          for (var i = 0; i < js_urls.length; i++) {
            var url = js_urls[i];
            var s = document.createElement('script');
            s.src = url;
            s.async = false;
            s.onreadystatechange = s.onload = function() {
              window._bokeh_is_loading--;
              if (window._bokeh_is_loading === 0) {
                console.log("Bokeh: all BokehJS libraries loaded");
                run_callbacks()
              }
            };
            s.onerror = function() {
              console.warn("failed to load library " + url);
            };
            console.log("Bokeh: injecting script tag for BokehJS library: ", url);
            document.getElementsByTagName("head")[0].appendChild(s);
          }
        };var element = document.getElementById("bbbf02e2-2946-41b4-8bb1-e1cd8277265a");
        if (element == null) {
          console.log("Bokeh: ERROR: autoload.js configured with elementid 'bbbf02e2-2946-41b4-8bb1-e1cd8277265a' but no matching script tag was found. ")
          return false;
        }
      
        var js_urls = [];
      
        var inline_js = [
          function(Bokeh) {
            Bokeh.$(function() {
                var docs_json = {"bf260907-8aea-413f-ad2a-41badba0755c":{"roots":{"references":[{"attributes":{},"id":"cf29c302-76bc-4fb2-aa91-91b48c2117b3","type":"ToolEvents"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":20},"x":{"field":"x"},"y":{"field":"y"}},"id":"e6f96007-96f5-44bc-8c79-8cf25f1702cc","type":"Circle"},{"attributes":{"fill_alpha":{"value":0.5},"fill_color":{"value":"yellow"},"line_alpha":{"value":0.5},"line_color":{"value":"yellow"},"size":{"units":"screen","value":20},"x":{"field":"x"},"y":{"field":"y"}},"id":"87471b9e-0866-4590-9d18-597360f3f22f","type":"Circle"},{"attributes":{"data_source":{"id":"b4ed43e5-02c4-4d80-b540-c36e1fd67932","type":"ColumnDataSource"},"glyph":{"id":"9b34501d-c353-4884-b166-2a71cd1b1e38","type":"Circle"},"hover_glyph":null,"nonselection_glyph":{"id":"fb30b46d-06a5-493e-9d2c-f99b5080a825","type":"Circle"},"selection_glyph":null},"id":"6d57e398-bd4f-40f4-99ae-1405e570093f","type":"GlyphRenderer"},{"attributes":{"plot":null,"text":"Mouse over the dots"},"id":"a2d9a937-338d-41d0-8e0b-23f3bcc1e493","type":"Title"},{"attributes":{"callback":null,"column_names":["imgs","y","desc","x"],"data":{"desc":["0","0","0","0","0","0","0","0","0","0","0","0","0","1","1","1","1","1","1","1","1","1","1","1","1","1","0","0","11","11","13","13","14","14","14","1","1","1","1","2","2","2","3","3","3","4","4","5","5","7","7","7","8","8","8"],"imgs":["./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_0_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_0_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_0_2A_10.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_0_2A_11.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_0_2A_12.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_0_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_0_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_0_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_0_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_0_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_0_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_0_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_0_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_1_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_1_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_1_2A_10.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_1_2A_11.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_1_2A_12.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_1_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_1_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_1_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_1_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_1_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_1_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_1_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c0/CL0401_Recombination_1_2A_9.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_0_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_0_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_11_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_11_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_13_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_13_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_14_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_14_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_14_6A_2.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_1_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_1_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_1_6A_2.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_1_6A_3.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_2_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_2_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_2_6A_2.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_3_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_3_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_3_6A_2.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_4_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_4_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_5_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_5_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_7_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_7_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_7_6A_2.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_8_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_8_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c0/CL0401_Recombination_8_6A_2.jpg"],"x":[-16.93059730420838,-16.306903857637895,-15.905685212308999,-15.438513115189654,-14.746017059409514,-16.51356855817101,-16.37897117760135,-15.559637633617223,-15.456157675326413,-16.653827973563082,-16.4371480248784,-15.577405264002952,-15.541348832485218,-12.74110131466704,-12.728941346474175,-11.320657279592071,-11.018139113280046,-10.708248710104572,-12.797097722477542,-12.68961510617604,-12.449719795721272,-11.910572005618404,-11.564149504196582,-11.621184832057994,-10.890482170387015,-11.20131359890657,-26.333176474541855,-26.339830368973413,-28.344029908039406,-28.147754412125643,-26.295727181597133,-26.41003816354378,-25.97340825556209,-25.87859225826994,-26.18419919073135,-24.921360335391384,-24.640540436394577,-24.65667379738513,-24.525761946384847,-28.829347654862605,-28.82010728099268,-28.828701565975116,-29.378880585023154,-29.688584107335707,-29.763015844643324,-27.300134478531945,-27.158662254149867,-29.01801633374179,-28.833245643120826,-27.674748880502968,-27.848217503121838,-27.39704021410868,-27.24581298431208,-26.877119686372595,-26.813704950055964],"y":[1.5674339499143999,1.6944736904343094,0.1621985855897923,-0.6369320899292659,0.28671698421902,1.6656363181169467,1.4031003471086254,1.0640974851656682,0.6920586427312253,-1.470586253602159,-1.294368584162195,-0.7516350618375914,-0.7540837148516318,0.8069505799689833,0.6129422704854094,0.4938858098903132,0.6205175220831742,0.4937665066758538,-0.8122182066779419,-0.19691268604912435,0.3223016860740722,-0.1779080217541197,-0.09701230428283968,-0.7408500983590898,-0.5953503896240125,-0.13579531745291462,-14.97747183809892,-14.959940580565695,-6.071598561578549,-6.263909093936683,-5.672402645047215,-5.685730014192343,-6.753198364571324,-7.010408685794095,-7.006871591070978,-11.958038758700166,-11.516231233244108,-11.39562879454511,-11.057280487042142,-10.041081529301403,-10.072083918278377,-10.033614931547119,-7.314993957995701,-9.302835620155914,-9.053643838086673,-6.356191800005329,-6.497991955543555,-6.8344572434926425,-6.912285455698143,-5.443575415784916,-5.345132279621356,-5.506049879275349,-7.362007050990382,-7.7114552458772,-7.918305920297899]}},"id":"b4ed43e5-02c4-4d80-b540-c36e1fd67932","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"d2994064-b778-43a3-8456-0b578dc96461","type":"ColumnDataSource"},"glyph":{"id":"87471b9e-0866-4590-9d18-597360f3f22f","type":"Circle"},"hover_glyph":null,"nonselection_glyph":{"id":"e6f96007-96f5-44bc-8c79-8cf25f1702cc","type":"Circle"},"selection_glyph":null},"id":"0fa7e578-2a80-48f6-ad01-45b1d0aec379","type":"GlyphRenderer"},{"attributes":{"dimension":1,"plot":{"id":"e0df0f74-b5b4-49d0-9f85-67dbfc01340d","subtype":"Figure","type":"Plot"},"ticker":{"id":"f6966f63-fff6-4fb8-90e7-0e26e65f3648","type":"BasicTicker"}},"id":"92000507-e361-474a-bc9e-52ea10fc8f86","type":"Grid"},{"attributes":{"callback":null,"column_names":["imgs","y","desc","x"],"data":{"desc":["F3Y64830U17GY4TA","F3Y64830U17GY4TA","F3Y64830U17GY4TA","F3Y64960NGVGY4TA","F3Y64960NGVGY4TA","F3Y64960NGVGY4TA","F3Y64960NGVGY4TA","F3Y64960NGVGY4TA","F3Y65020Q5ZGY4TA","F3Y65020Q5ZGY4TA","F3Y65020Q5ZGY4TA","F3Y65020Q5ZGY4TA","F3Y65020Q5ZGY4TA","F3Y65020Q5ZGY4TA","F3Y65020Q5ZGY4TA","F3Y65020Q5ZGY4TA","F3Y65020Q5ZGY4TA","F3Y65020Q5ZGY4TA","F3Y65020Q5ZGY4TA","F3Y65020RGDGY4RA","F3Y65030LJFGY4RA","F3Y65030LJFGY4RA","F3Y65030LJFGY4RA","F3Y65030LJFGY4RA","F3Y65030LJFGY4RA","F3Y65030LJFGY4RA","F3Y65030LJFGY4RA","F3Y65030LJFGY4RA","F3Y65030LJFGY4RA","F3Y65030LJFGY4RA","F3Y65030LJFGY4RA","F3Y65030LK5GY4RA","F3Y65030LK5GY4RA","F3Y65030LK5GY4RA","F3Y65030LK5GY4RA","F3Y65030LK5GY4RA","F3Y65030LK5GY4RA","F3Y65030LK5GY4RA","F3Y65030LK5GY4RA","F3Y65030LK5GY4RA","F3Y65030LK5GY4RA","F3Y65030LNGGY4RA","F3Y65030LNGGY4RA","F3Y65030LNGGY4RA","F3Y65030LWZGY4RA","F3Y65030LWZGY4RA","F3Y65030LWZGY4RA","F3Y65030LWZGY4RA","F3Y65030LWZGY4RA","F3Y65030LWZGY4RA","F3Y65030LWZGY4RA","F3Y65030LWZGY4RA","F3Y65030MV5GY4RA","F3Y65030MV5GY4RA","F3Y650408AWGY4RA","F3Y650408AWGY4RA","F3Y650408AWGY4RA","F3Y650408AWGY4RA","F3Y650408AWGY4RA","F3Y65040TQ2GY4RA","F3Y64830U17GY4TA","F3Y64910MN9GY4TA","F3Y64912CK9GY4TA","F3Y64960NGVGY4TA","F3Y65020Q5ZGY4TA","F3Y65020RGDGY4RA","F3Y65020RGEGY4RA","F3Y65030LJFGY4RA","F3Y65030LK5GY4RA","F3Y65030LNGGY4RA","F3Y65030LWZGY4RA","F3Y65030MV5GY4RA","F3Y650408AWGY4RA","F3Y65040ACYGY4RA","F3Y65040AQUGY4RA","F3Y65050B05GY4RB","F3Y65050BKZGY4RB","F3Y65050FSAGY4RB","F3Y65050PZTGY4RB"],"imgs":["./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y64830U17GY4TA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y64830U17GY4TA_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y64830U17GY4TA_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y64960NGVGY4TA_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y64960NGVGY4TA_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y64960NGVGY4TA_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y64960NGVGY4TA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y64960NGVGY4TA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65020Q5ZGY4TA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65020Q5ZGY4TA_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65020Q5ZGY4TA_2A_10.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65020Q5ZGY4TA_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65020Q5ZGY4TA_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65020Q5ZGY4TA_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65020Q5ZGY4TA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65020Q5ZGY4TA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65020Q5ZGY4TA_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65020Q5ZGY4TA_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65020Q5ZGY4TA_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65020RGDGY4RA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LJFGY4RA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LJFGY4RA_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LJFGY4RA_2A_10.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LJFGY4RA_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LJFGY4RA_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LJFGY4RA_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LJFGY4RA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LJFGY4RA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LJFGY4RA_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LJFGY4RA_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LJFGY4RA_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LK5GY4RA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LK5GY4RA_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LK5GY4RA_2A_10.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LK5GY4RA_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LK5GY4RA_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LK5GY4RA_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LK5GY4RA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LK5GY4RA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LK5GY4RA_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LK5GY4RA_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LNGGY4RA_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LNGGY4RA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LNGGY4RA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LWZGY4RA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LWZGY4RA_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LWZGY4RA_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LWZGY4RA_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LWZGY4RA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LWZGY4RA_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LWZGY4RA_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030LWZGY4RA_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030MV5GY4RA_2A_18.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65030MV5GY4RA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y650408AWGY4RA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y650408AWGY4RA_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y650408AWGY4RA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y650408AWGY4RA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y650408AWGY4RA_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c1/CL0401_Recombination_F3Y65040TQ2GY4RA_2A_11.jpg","./ben1214/CL0401/8A/CL0401_8A_c1/CL0401_Recombination_F3Y64830U17GY4TA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c1/CL0401_Recombination_F3Y64910MN9GY4TA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c1/CL0401_Recombination_F3Y64912CK9GY4TA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c1/CL0401_Recombination_F3Y64960NGVGY4TA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c1/CL0401_Recombination_F3Y65020Q5ZGY4TA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c1/CL0401_Recombination_F3Y65020RGDGY4RA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c1/CL0401_Recombination_F3Y65020RGEGY4RA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c1/CL0401_Recombination_F3Y65030LJFGY4RA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c1/CL0401_Recombination_F3Y65030LK5GY4RA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c1/CL0401_Recombination_F3Y65030LNGGY4RA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c1/CL0401_Recombination_F3Y65030LWZGY4RA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c1/CL0401_Recombination_F3Y65030MV5GY4RA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c1/CL0401_Recombination_F3Y650408AWGY4RA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c1/CL0401_Recombination_F3Y65040ACYGY4RA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c1/CL0401_Recombination_F3Y65040AQUGY4RA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c1/CL0401_Recombination_F3Y65050B05GY4RB_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c1/CL0401_Recombination_F3Y65050BKZGY4RB_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c1/CL0401_Recombination_F3Y65050FSAGY4RB_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c1/CL0401_Recombination_F3Y65050PZTGY4RB_8A_0.jpg"],"x":[-16.885248840498193,-16.939710551533445,-17.296854866238082,-16.28545026566499,-16.451330124925565,-16.27193869378673,-16.237729293341918,-16.009975102328248,-9.770645237188052,-9.482850074195953,-11.458927888547077,-9.427004904017165,-9.718916364843974,-9.94801969090759,-10.172495907376858,-10.485657671508342,-11.041199777442994,-11.460790920366142,-11.52681606910583,-17.50730685307905,-18.65606867858495,-18.782103027551063,-18.491054927073254,-18.364691111606795,-17.92555274334966,-17.548098279944238,-17.468575627210004,-16.95635700496615,-16.505052150746494,-16.960001059196212,-17.20586601178209,-19.77523806582587,-19.87623366167967,-19.083524820787982,-20.494854816802402,-20.613166692239705,-20.55004969992167,-19.603424556496748,-19.839087556973574,-19.28665816915847,-19.25626820749136,-17.755030945922964,-17.66280490787628,-17.64912717773957,-10.525250867505466,-10.66832533539789,-10.274552709731262,-10.164549928524867,-9.775069136679734,-9.568263524065223,-9.480968592228953,-9.633481143384868,-16.38426326440386,-16.418044088935662,-14.359018749124003,-14.538957187506641,-14.765560366756358,-14.733839684341445,-14.787951283530793,-0.9318716367522754,5.36198921746922,6.292924387812055,6.143293755806932,10.280702544758869,5.989414140082113,10.323056264712294,10.937823621530029,5.6130544889600165,9.195407145858924,7.862490478424137,5.485909273607488,5.096266773447874,4.888822144823318,8.767919915567198,4.963692947525721,7.948003175473439,6.2074041451462225,4.979719855629517,6.654575518910866],"y":[17.137608315001792,17.19899909992633,17.378826297832845,12.016149887164831,12.359040629089096,12.90062855243908,12.897855295414622,13.059950704373904,5.0152529893468785,5.025213264592085,4.80848367813899,5.4495835522518155,5.788855462305431,5.957970719020203,5.747560856534729,5.94826476337612,5.374838644495532,5.123879196972069,4.805379521743037,12.063835236703905,19.419482998711644,19.31620059357226,20.798077632903095,19.438987926910496,19.841238357641892,20.262045020457048,20.671384726329872,20.287763387669163,19.86774280459876,19.729201223689547,19.759177741717735,13.46335534223456,13.366848165542756,14.998001638703375,13.400029865579103,14.017703610355307,14.63016639030685,14.413202366186777,14.791477022506527,14.835525037239552,15.042747448409443,12.96426013750316,13.579413695679763,13.652730543525996,11.379954782526127,11.325278231446186,11.463011129046393,11.425806871286198,12.469149259281046,12.776288322786815,12.893575303287502,12.76966712696584,15.008470785960391,15.246298101191474,6.762830602692015,6.423208641109228,6.143702596991993,6.178920698406133,5.920866351686529,19.072331136928696,-29.347868553383,-31.87267857052973,-29.708110704882866,-30.541310415211537,-31.341091734670176,-30.29528574009161,-28.95183637036821,-27.57491223857309,-29.486873141836714,-29.608717324377476,-30.240251039950365,-28.715518686859692,-30.588577244079747,-28.96225249799422,-30.234708676984145,-32.320256604364566,-30.91740085037712,-31.13589511698783,-31.3198369926536]}},"id":"f078f1e9-841e-431e-9db8-df298b13f107","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"value":0.5},"fill_color":{"value":"green"},"line_alpha":{"value":0.5},"line_color":{"value":"green"},"size":{"units":"screen","value":20},"x":{"field":"x"},"y":{"field":"y"}},"id":"9b34501d-c353-4884-b166-2a71cd1b1e38","type":"Circle"},{"attributes":{},"id":"06535a33-6b00-47df-8559-d96feff9cf07","type":"BasicTickFormatter"},{"attributes":{"callback":null,"column_names":["imgs","y","desc","x"],"data":{"desc":["F3Y649617UMH3RPA","F3Y650301LXH3RPA","F3Y65030D8BH3RPA","F3Y65030D8BH3RPA","F3Y65030MXNGY4RA","F3Y65030MXNGY4RA","F3Y650311ELH3RPA","F3Y650311ELH3RPA","F3Y65040NAAGY4TA","F3Y65040NAAGY4TA","F3Y65040NAAGY4TA","F3Y65040NAAGY4TA","F3Y65040SAPGY4RA","F3Y65040SAPGY4RA","F3Y65040SAPGY4RA","F3Y650503F4H3RQC","F3Y650503F4H3RQC","F3Y650503F4H3RQC","F3Y6505044EH3RQC","F3Y6505044EH3RQC","F3Y6505044EH3RQC","F3Y65050AXKGY4RB","F3Y65050BHLH3RQC","F3Y65050BHLH3RQC","F3Y65050BRGGY4RB","F3Y65050BRGGY4RB","F3Y65050BRGGY4RB","F3Y65050C5JH3RQC","F3Y65050C5JH3RQC","F3Y65050CT3H3RQC","F3Y65050CT3H3RQC","F3Y65050CT3H3RQC","F3Y65050FM7H3RPB","F3Y65050FMZGY4RB","F3Y65050FMZGY4RB","F3Y65050NYGH3RPB","F3Y65050PJSH3RQC","F3Y6505124AH3RPB","F3Y6505124AH3RPB","F3Y649617UMH3RPA","F3Y650301LXH3RPA","F3Y65030D8BH3RPA","F3Y650311ELH3RPA","F3Y65040NAAGY4TA","F3Y65040SAPGY4RA","F3Y650503F4H3RQC","F3Y6505044EH3RQC","F3Y65050AXKGY4RB","F3Y65050BHLH3RQC","F3Y65050BRGGY4RB","F3Y65050C5JH3RQC","F3Y65050FM7H3RPB","F3Y65050NYGH3RPB","F3Y6505124AH3RPB"],"imgs":["./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y649617UMH3RPA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y650301LXH3RPA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65030D8BH3RPA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65030D8BH3RPA_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65030MXNGY4RA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65030MXNGY4RA_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y650311ELH3RPA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y650311ELH3RPA_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65040NAAGY4TA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65040NAAGY4TA_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65040NAAGY4TA_6A_2.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65040NAAGY4TA_6A_3.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65040SAPGY4RA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65040SAPGY4RA_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65040SAPGY4RA_6A_2.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y650503F4H3RQC_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y650503F4H3RQC_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y650503F4H3RQC_6A_2.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y6505044EH3RQC_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y6505044EH3RQC_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y6505044EH3RQC_6A_2.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65050AXKGY4RB_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65050BHLH3RQC_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65050BHLH3RQC_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65050BRGGY4RB_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65050BRGGY4RB_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65050BRGGY4RB_6A_2.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65050C5JH3RQC_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65050C5JH3RQC_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65050CT3H3RQC_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65050CT3H3RQC_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65050CT3H3RQC_6A_2.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65050FM7H3RPB_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65050FMZGY4RB_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65050FMZGY4RB_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65050NYGH3RPB_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y65050PJSH3RQC_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y6505124AH3RPB_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c7/CL0401_Recombination_F3Y6505124AH3RPB_6A_1.jpg","./ben1214/CL0401/8A/CL0401_8A_c7/CL0401_Recombination_F3Y649617UMH3RPA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c7/CL0401_Recombination_F3Y650301LXH3RPA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c7/CL0401_Recombination_F3Y65030D8BH3RPA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c7/CL0401_Recombination_F3Y650311ELH3RPA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c7/CL0401_Recombination_F3Y65040NAAGY4TA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c7/CL0401_Recombination_F3Y65040SAPGY4RA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c7/CL0401_Recombination_F3Y650503F4H3RQC_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c7/CL0401_Recombination_F3Y6505044EH3RQC_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c7/CL0401_Recombination_F3Y65050AXKGY4RB_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c7/CL0401_Recombination_F3Y65050BHLH3RQC_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c7/CL0401_Recombination_F3Y65050BRGGY4RB_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c7/CL0401_Recombination_F3Y65050C5JH3RQC_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c7/CL0401_Recombination_F3Y65050FM7H3RPB_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c7/CL0401_Recombination_F3Y65050NYGH3RPB_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c7/CL0401_Recombination_F3Y6505124AH3RPB_8A_0.jpg"],"x":[-23.164496808040223,-22.02357179280082,-21.118805170050628,-21.130521614905614,-22.821434092335956,-22.720318166604738,-19.136313127091032,-19.108805516759347,-18.4019763315407,-18.595716954290978,-18.66244118082056,-18.937042817459407,-20.88888714908329,-20.908616176695343,-20.852748790065956,-24.127104847384093,-24.042985999281004,-23.875400832043933,-21.39966108509161,-21.198861532877253,-21.13502683369031,-23.916272164401743,-18.67348611598106,-18.645907455569972,-20.092833700175483,-20.022066487376836,-19.74446934942087,-20.43158453264333,-20.926796668321334,-24.759507520122924,-24.713528106607498,-24.792760319291506,-19.64668217022929,-20.41090590340292,-20.386474152234314,-23.398052151295243,-18.941771127935603,-17.666810989558133,-17.83236544310018,9.42865417277347,8.30537620150793,6.816380506995044,6.7955153989447625,8.579628345932,8.814556704901731,7.140497569648269,8.55798769656667,9.513402438468585,8.503618735724624,8.529699455616738,6.885709294794047,8.588335714919799,8.116979259924582,9.937322855226714],"y":[-18.0961873531898,-18.187090148299,-14.018909512763688,-14.07272450155686,-12.64029480190966,-12.748481841731724,-18.147380541913005,-18.12095445990431,-15.136625998318912,-15.452068629876226,-15.732860178900701,-15.59766245372809,-11.401626718280628,-11.417423980310787,-11.464244346244838,-16.479536587474573,-16.613907936964353,-16.56609555776619,-15.617538144645152,-15.463479459785857,-15.723789417488367,-15.683396319035936,-12.44659697843228,-12.34992092237948,-14.888822273646612,-14.727436923427522,-14.2624948985943,-15.329842637319327,-17.054922966370636,-18.216497328008042,-18.207591306388622,-18.208491786349576,-16.308049161246426,-17.723555866482293,-17.80764896544359,-15.915294636291508,-16.990096253641404,-16.81315879990667,-16.807945018311845,-27.070984310477503,-27.43419652130027,-28.426057659398786,-29.212669168976994,-28.375999479976123,-30.096120803771953,-27.531603032335227,-29.16800756145474,-28.699816643567278,-31.053994237266476,-29.50598389345762,-28.38536692182567,-27.385763146049385,-28.865082312731445,-29.179410673437523]}},"id":"d2994064-b778-43a3-8456-0b578dc96461","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"value":0.5},"fill_color":{"value":"red"},"line_alpha":{"value":0.5},"line_color":{"value":"red"},"size":{"units":"screen","value":20},"x":{"field":"x"},"y":{"field":"y"}},"id":"41eb632a-f3d0-47e8-a481-501b77d8be1b","type":"Circle"},{"attributes":{"formatter":{"id":"27a8e47b-1748-4f4e-b153-09cd26d13506","type":"BasicTickFormatter"},"plot":{"id":"e0df0f74-b5b4-49d0-9f85-67dbfc01340d","subtype":"Figure","type":"Plot"},"ticker":{"id":"14a805c0-0707-44f3-82b7-49deb04fc6e6","type":"BasicTicker"}},"id":"3473d385-9348-47d5-bf4c-f690685ec729","type":"LinearAxis"},{"attributes":{},"id":"27a8e47b-1748-4f4e-b153-09cd26d13506","type":"BasicTickFormatter"},{"attributes":{"callback":null,"plot":{"id":"e0df0f74-b5b4-49d0-9f85-67dbfc01340d","subtype":"Figure","type":"Plot"},"tooltips":"\n                <div>\n                    <div>\n                        <img\n                            src=\"@imgs\" height=\"80\" alt=\"@imgs\" width=\"80\"\n                            style=\"float: left; margin: 0px 15px 15px 0px;\"\n                            border=\"2\"\n                        ></img>\n                    </div>\n                    <div>\n                        <span style=\"font-size: 17px; font-weight: bold;\">@desc</span>\n                    </div>\n                </div>\n            "},"id":"c7a7d445-3337-4387-9f83-d67c72c135ae","type":"HoverTool"},{"attributes":{"callback":null,"column_names":["imgs","y","desc","x"],"data":{"desc":["F3Y6452037UGY4TA","F3Y6452037UGY4TA","F3Y6452037UGY4TA","F3Y6452037UGY4TA","F3Y6452037UGY4TA","F3Y6452037UGY4TA","F3Y6452037UGY4TA","F3Y6452037UGY4TA","F3Y6452037UGY4TA","F3Y6452037UGY4TA","F3Y6452037UGY4TA","F3Y647219XDH3RNA","F3Y647219XDH3RNA","F3Y647219XDH3RNA","F3Y647219XDH3RNA","F3Y647219XDH3RNA","F3Y647219XDH3RNA","F3Y647219XDH3RNA","F3Y647219XDH3RNA","F3Y647219XDH3RNA","F3Y647219XDH3RNA","F3Y647219XDH3RNA","F3Y64760V7WH3RNA","F3Y64760V7WH3RNA","F3Y64760V7WH3RNA","F3Y64760V7WH3RNA","F3Y64760V7WH3RNA","F3Y64760V7WH3RNA","F3Y64760V7WH3RNA","F3Y64760V7WH3RNA","F3Y64760V7WH3RNA","F3Y64760V7WH3RNA","F3Y6491000FGY4TA","F3Y6491000FGY4TA","F3Y6491000FGY4TA","F3Y6491000FGY4TA","F3Y6491000FGY4TA","F3Y6491000FGY4TA","F3Y6491000FGY4TA","F3Y6491000FGY4TA","F3Y6491000FGY4TA","F3Y64910ZSFGY4TA","F3Y64910ZSFGY4TA","F3Y64910ZSFGY4TA","F3Y64910ZSFGY4TA","F3Y64910ZSFGY4TA","F3Y64910ZSFGY4TA","F3Y64910ZSFGY4TA","F3Y64910ZSFGY4TA","F3Y64910ZSFGY4TA","F3Y64910ZSFGY4TA","F3Y64910ZSFGY4TA","F3Y64910ZSFGY4TA","F3Y64920UCWGY4TA","F3Y64920UCWGY4TA","F3Y64920UCWGY4TA","F3Y64920UCWGY4TA","F3Y64920UCWGY4TA","F3Y64920UCWGY4TA","F3Y64920UCWGY4TA","F3Y64920UCWGY4TA","F3Y64920UCWGY4TA","F3Y64920UCWGY4TA","F3Y64920UCWGY4TA","F3Y649504L0GY4TA","F3Y649504L0GY4TA","F3Y649504L0GY4TA","F3Y649504L0GY4TA","F3Y649504L0GY4TA","F3Y649504L0GY4TA","F3Y649504L0GY4TA","F3Y649504L0GY4TA","F3Y649504L0GY4TA","F3Y649504L0GY4TA","F3Y649504L0GY4TA","F3Y64950V9FH3RPA","F3Y64950V9FH3RPA","F3Y64950V9FH3RPA","F3Y64950V9FH3RPA","F3Y64950V9FH3RPA","F3Y64950V9FH3RPA","F3Y64950V9FH3RPA","F3Y64950V9FH3RPA","F3Y64950V9FH3RPA","F3Y64950V9FH3RPA","F3Y6501083MGY4TA","F3Y6501083MGY4TA","F3Y6501083MGY4TA","F3Y6501083MGY4TA","F3Y6501083MGY4TA","F3Y6501083MGY4TA","F3Y6501083MGY4TA","F3Y6501083MGY4TA","F3Y6501083MGY4TA","F3Y6501083MGY4TA","F3Y65013JQKGY4TA","F3Y65013JQKGY4TA","F3Y65013JQKGY4TA","F3Y65013JQKGY4TA","F3Y65013JQKGY4TA","F3Y65013JQKGY4TA","F3Y65013JQKGY4TA","F3Y65013JQKGY4TA","F3Y65013JQKGY4TA","F3Y65013JQKGY4TA","F3Y65020P8TGY4TA","F3Y65020P8TGY4TA","F3Y65020P8TGY4TA","F3Y65020P8TGY4TA","F3Y65020P8TGY4TA","F3Y65020P8TGY4TA","F3Y65020P8TGY4TA","F3Y65020P8TGY4TA","F3Y65020P8TGY4TA","F3Y65020P8TGY4TA","F3Y65030YU5GY4TA","F3Y65030YU5GY4TA","F3Y65030YU5GY4TA","F3Y65030YU5GY4TA","F3Y65030YU5GY4TA","F3Y65030YU5GY4TA","F3Y65030YU5GY4TA","F3Y65030YU5GY4TA","F3Y65030YU5GY4TA","F3Y65030YU5GY4TA","F3Y65030YU5GY4TA","F3Y650403D2H3RPA","F3Y650403D2H3RPA","F3Y650403D2H3RPA","F3Y650403D2H3RPA","F3Y650403D2H3RPA","F3Y650403D2H3RPA","F3Y650403D2H3RPA","F3Y650403D2H3RPA","F3Y650403D2H3RPA","F3Y650403D2H3RPA","F3Y650408E2GY4RA","F3Y650408E2GY4RA","F3Y650408E2GY4RA","F3Y650408E2GY4RA","F3Y650408E2GY4RA","F3Y650408E2GY4RA","F3Y650408E2GY4RA","F3Y650408E2GY4RA","F3Y650408E2GY4RA","F3Y650408E2GY4RA","F3Y65040ST2GY4RA","F3Y65040ST2GY4RA","F3Y65040ST2GY4RA","F3Y65040ST2GY4RA","F3Y65040ST2GY4RA","F3Y65040ST2GY4RA","F3Y65040ST2GY4RA","F3Y65040ST2GY4RA","F3Y65040ST2GY4RA","F3Y65040ST2GY4RA","F3Y65040ST2GY4RA","F3Y65050B9NH3RQC","F3Y65050B9NH3RQC","F3Y65050B9NH3RQC","F3Y65050B9NH3RQC","F3Y65050B9NH3RQC","F3Y65050B9NH3RQC","F3Y65050B9NH3RQC","F3Y65050B9NH3RQC","F3Y65050B9NH3RQC","F3Y65050B9NH3RQC","F3Y65050EF8GY4TB","F3Y65050EF8GY4TB","F3Y65050EF8GY4TB","F3Y65050EF8GY4TB","F3Y65050EF8GY4TB","F3Y65050EF8GY4TB","F3Y65050EF8GY4TB","F3Y65050EF8GY4TB","F3Y65050EF8GY4TB","F3Y65050EF8GY4TB","F3Y65050FNZGY4TB","F3Y65050FNZGY4TB","F3Y65050FNZGY4TB","F3Y65050FNZGY4TB","F3Y65050FNZGY4TB","F3Y65050FNZGY4TB","F3Y65050FNZGY4TB","F3Y65050FNZGY4TB","F3Y65050FNZGY4TB","F3Y65050FNZGY4TB","F3Y65050FX3GY4RB","F3Y65050FX3GY4RB","F3Y65050FX3GY4RB","F3Y65050FX3GY4RB","F3Y65050FX3GY4RB","F3Y65050FX3GY4RB","F3Y65050FX3GY4RB","F3Y65050FX3GY4RB","F3Y65050FX3GY4RB","F3Y65050FX3GY4RB","F3Y65050P7DGY4RB","F3Y65050P7DGY4RB","F3Y65050P7DGY4RB","F3Y65050P7DGY4RB","F3Y65050P7DGY4RB","F3Y65050P7DGY4RB","F3Y65050P7DGY4RB","F3Y65050P7DGY4RB","F3Y65050P7DGY4RB","F3Y65050P7DGY4RB","F3Y6452037UGY4TA","F3Y647219XDH3RNA","F3Y64760V7WH3RNA","F3Y64760V7WH3RNA","F3Y64760V7WH3RNA","F3Y6491000FGY4TA","F3Y6491000FGY4TA","F3Y64910ZSFGY4TA","F3Y64910ZSFGY4TA","F3Y64910ZSFGY4TA","F3Y64910ZSFGY4TA","F3Y64920UCWGY4TA","F3Y64920UCWGY4TA","F3Y64920UCWGY4TA","F3Y649504L0GY4TA","F3Y649504L0GY4TA","F3Y64950V9FH3RPA","F3Y64950V9FH3RPA","F3Y64950V9FH3RPA","F3Y6501083MGY4TA","F3Y65013JQKGY4TA","F3Y65020P8TGY4TA","F3Y65030YU5GY4TA","F3Y65030YU5GY4TA","F3Y650403D2H3RPA","F3Y650403D2H3RPA","F3Y650408E2GY4RA","F3Y650408E2GY4RA","F3Y65040ST2GY4RA","F3Y65050B9NH3RQC","F3Y65050B9NH3RQC","F3Y65050EF8GY4TB","F3Y65050FNZGY4TB","F3Y65050FX3GY4RB","F3Y65050P7DGY4RB","F3Y65050P7DGY4RB","F3Y6452037UGY4TA","F3Y647219XDH3RNA","F3Y64760V7WH3RNA","F3Y6491000FGY4TA","F3Y64910ZSFGY4TA","F3Y64920UCWGY4TA","F3Y649504L0GY4TA","F3Y64950V9FH3RPA","F3Y6501083MGY4TA","F3Y65013JQKGY4TA","F3Y65020P8TGY4TA","F3Y65030YU5GY4TA","F3Y65040ST2GY4RA","F3Y65050B9NH3RQC","F3Y65050EF8GY4TB","F3Y65050FNZGY4TB","F3Y65050P7DGY4RB"],"imgs":["./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6452037UGY4TA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6452037UGY4TA_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6452037UGY4TA_2A_10.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6452037UGY4TA_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6452037UGY4TA_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6452037UGY4TA_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6452037UGY4TA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6452037UGY4TA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6452037UGY4TA_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6452037UGY4TA_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6452037UGY4TA_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y647219XDH3RNA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y647219XDH3RNA_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y647219XDH3RNA_2A_10.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y647219XDH3RNA_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y647219XDH3RNA_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y647219XDH3RNA_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y647219XDH3RNA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y647219XDH3RNA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y647219XDH3RNA_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y647219XDH3RNA_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y647219XDH3RNA_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64760V7WH3RNA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64760V7WH3RNA_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64760V7WH3RNA_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64760V7WH3RNA_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64760V7WH3RNA_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64760V7WH3RNA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64760V7WH3RNA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64760V7WH3RNA_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64760V7WH3RNA_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64760V7WH3RNA_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6491000FGY4TA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6491000FGY4TA_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6491000FGY4TA_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6491000FGY4TA_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6491000FGY4TA_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6491000FGY4TA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6491000FGY4TA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6491000FGY4TA_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6491000FGY4TA_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64910ZSFGY4TA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64910ZSFGY4TA_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64910ZSFGY4TA_2A_10.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64910ZSFGY4TA_2A_11.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64910ZSFGY4TA_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64910ZSFGY4TA_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64910ZSFGY4TA_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64910ZSFGY4TA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64910ZSFGY4TA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64910ZSFGY4TA_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64910ZSFGY4TA_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64910ZSFGY4TA_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64920UCWGY4TA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64920UCWGY4TA_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64920UCWGY4TA_2A_10.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64920UCWGY4TA_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64920UCWGY4TA_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64920UCWGY4TA_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64920UCWGY4TA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64920UCWGY4TA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64920UCWGY4TA_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64920UCWGY4TA_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64920UCWGY4TA_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y649504L0GY4TA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y649504L0GY4TA_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y649504L0GY4TA_2A_10.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y649504L0GY4TA_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y649504L0GY4TA_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y649504L0GY4TA_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y649504L0GY4TA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y649504L0GY4TA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y649504L0GY4TA_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y649504L0GY4TA_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y649504L0GY4TA_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64950V9FH3RPA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64950V9FH3RPA_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64950V9FH3RPA_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64950V9FH3RPA_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64950V9FH3RPA_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64950V9FH3RPA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64950V9FH3RPA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64950V9FH3RPA_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64950V9FH3RPA_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y64950V9FH3RPA_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6501083MGY4TA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6501083MGY4TA_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6501083MGY4TA_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6501083MGY4TA_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6501083MGY4TA_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6501083MGY4TA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6501083MGY4TA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6501083MGY4TA_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6501083MGY4TA_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y6501083MGY4TA_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65013JQKGY4TA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65013JQKGY4TA_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65013JQKGY4TA_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65013JQKGY4TA_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65013JQKGY4TA_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65013JQKGY4TA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65013JQKGY4TA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65013JQKGY4TA_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65013JQKGY4TA_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65013JQKGY4TA_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65020P8TGY4TA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65020P8TGY4TA_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65020P8TGY4TA_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65020P8TGY4TA_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65020P8TGY4TA_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65020P8TGY4TA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65020P8TGY4TA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65020P8TGY4TA_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65020P8TGY4TA_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65020P8TGY4TA_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65030YU5GY4TA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65030YU5GY4TA_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65030YU5GY4TA_2A_10.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65030YU5GY4TA_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65030YU5GY4TA_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65030YU5GY4TA_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65030YU5GY4TA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65030YU5GY4TA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65030YU5GY4TA_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65030YU5GY4TA_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65030YU5GY4TA_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y650403D2H3RPA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y650403D2H3RPA_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y650403D2H3RPA_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y650403D2H3RPA_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y650403D2H3RPA_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y650403D2H3RPA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y650403D2H3RPA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y650403D2H3RPA_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y650403D2H3RPA_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y650403D2H3RPA_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y650408E2GY4RA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y650408E2GY4RA_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y650408E2GY4RA_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y650408E2GY4RA_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y650408E2GY4RA_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y650408E2GY4RA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y650408E2GY4RA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y650408E2GY4RA_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y650408E2GY4RA_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y650408E2GY4RA_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65040ST2GY4RA_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65040ST2GY4RA_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65040ST2GY4RA_2A_10.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65040ST2GY4RA_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65040ST2GY4RA_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65040ST2GY4RA_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65040ST2GY4RA_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65040ST2GY4RA_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65040ST2GY4RA_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65040ST2GY4RA_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65040ST2GY4RA_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050B9NH3RQC_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050B9NH3RQC_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050B9NH3RQC_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050B9NH3RQC_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050B9NH3RQC_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050B9NH3RQC_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050B9NH3RQC_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050B9NH3RQC_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050B9NH3RQC_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050B9NH3RQC_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050EF8GY4TB_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050EF8GY4TB_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050EF8GY4TB_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050EF8GY4TB_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050EF8GY4TB_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050EF8GY4TB_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050EF8GY4TB_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050EF8GY4TB_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050EF8GY4TB_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050EF8GY4TB_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050FNZGY4TB_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050FNZGY4TB_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050FNZGY4TB_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050FNZGY4TB_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050FNZGY4TB_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050FNZGY4TB_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050FNZGY4TB_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050FNZGY4TB_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050FNZGY4TB_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050FNZGY4TB_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050FX3GY4RB_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050FX3GY4RB_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050FX3GY4RB_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050FX3GY4RB_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050FX3GY4RB_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050FX3GY4RB_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050FX3GY4RB_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050FX3GY4RB_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050FX3GY4RB_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050FX3GY4RB_2A_9.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050P7DGY4RB_2A_0.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050P7DGY4RB_2A_1.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050P7DGY4RB_2A_2.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050P7DGY4RB_2A_3.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050P7DGY4RB_2A_4.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050P7DGY4RB_2A_5.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050P7DGY4RB_2A_6.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050P7DGY4RB_2A_7.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050P7DGY4RB_2A_8.jpg","./ben1214/CL0401/2A/CL0401_2A_c4/CL0401_Recombination_F3Y65050P7DGY4RB_2A_9.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y6452037UGY4TA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y647219XDH3RNA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y64760V7WH3RNA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y64760V7WH3RNA_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y64760V7WH3RNA_6A_2.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y6491000FGY4TA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y6491000FGY4TA_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y64910ZSFGY4TA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y64910ZSFGY4TA_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y64910ZSFGY4TA_6A_2.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y64910ZSFGY4TA_6A_3.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y64920UCWGY4TA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y64920UCWGY4TA_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y64920UCWGY4TA_6A_2.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y649504L0GY4TA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y649504L0GY4TA_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y64950V9FH3RPA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y64950V9FH3RPA_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y64950V9FH3RPA_6A_2.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y6501083MGY4TA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y65013JQKGY4TA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y65020P8TGY4TA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y65030YU5GY4TA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y65030YU5GY4TA_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y650403D2H3RPA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y650403D2H3RPA_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y650408E2GY4RA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y650408E2GY4RA_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y65040ST2GY4RA_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y65050B9NH3RQC_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y65050B9NH3RQC_6A_1.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y65050EF8GY4TB_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y65050FNZGY4TB_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y65050FX3GY4RB_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y65050P7DGY4RB_6A_0.jpg","./ben1214/CL0401/6A/CL0401_6A_c4/CL0401_Recombination_F3Y65050P7DGY4RB_6A_1.jpg","./ben1214/CL0401/8A/CL0401_8A_c4/CL0401_Recombination_F3Y6452037UGY4TA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c4/CL0401_Recombination_F3Y647219XDH3RNA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c4/CL0401_Recombination_F3Y64760V7WH3RNA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c4/CL0401_Recombination_F3Y6491000FGY4TA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c4/CL0401_Recombination_F3Y64910ZSFGY4TA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c4/CL0401_Recombination_F3Y64920UCWGY4TA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c4/CL0401_Recombination_F3Y649504L0GY4TA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c4/CL0401_Recombination_F3Y64950V9FH3RPA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c4/CL0401_Recombination_F3Y6501083MGY4TA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c4/CL0401_Recombination_F3Y65013JQKGY4TA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c4/CL0401_Recombination_F3Y65020P8TGY4TA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c4/CL0401_Recombination_F3Y65030YU5GY4TA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c4/CL0401_Recombination_F3Y65040ST2GY4RA_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c4/CL0401_Recombination_F3Y65050B9NH3RQC_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c4/CL0401_Recombination_F3Y65050EF8GY4TB_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c4/CL0401_Recombination_F3Y65050FNZGY4TB_8A_0.jpg","./ben1214/CL0401/8A/CL0401_8A_c4/CL0401_Recombination_F3Y65050P7DGY4RB_8A_0.jpg"],"x":[10.774277407863407,10.720585837986366,13.38775961003759,11.149257585886135,11.00760089749791,11.910290410816147,11.41794272881523,12.828221497934113,13.008887329108115,13.15838016728259,13.33282473484719,4.849930934259639,4.660810615503259,2.8506691714343884,4.255323172568812,3.8472270305778444,3.3643719448589438,2.843141778374623,2.675961421077575,2.1550884750332995,1.949934115187702,2.085684100685695,6.419624444894678,6.588995674663509,7.220436478801401,7.087664398705894,7.19352491374828,9.13462570760225,9.15065453199356,8.340771625572216,8.327205739937915,8.526815502298504,4.706447310744869,4.623046868989957,4.2990194957563,4.076066198444727,2.5717104891554206,1.8274701124703465,2.2091472687251246,1.1800325255663986,3.119858018765899,10.013257414471022,10.313150568727757,12.915326356036866,13.127012393674795,10.799703022490577,11.276788327827092,12.941623605548306,13.076371508510366,12.79451841943299,12.076383987212008,12.508657042090592,12.883309110448149,13.65460763464795,13.619584501639705,15.500326028572715,13.709326529740359,14.218788965566816,14.741174984519857,14.664144427437217,14.939269836613912,15.191684979355216,15.114614509822044,15.180167408035139,24.093624416390185,23.86728505961087,23.09802533196735,23.766674324552763,23.340091512161294,23.022247096210823,22.004681377261914,22.251825933963513,22.116022485396492,21.6507154387498,21.74886954909971,11.151822499465453,11.293494612129294,11.233901099190797,12.237995879335008,11.769293283464572,11.506961389287273,11.980683079313833,11.411233665572134,12.027435053553955,10.655597236100535,24.943322687169143,25.39844256462981,24.59876076806172,24.88585752770779,25.588101168753454,25.607651058054888,25.905314589535006,26.276282912511377,26.451817615262772,26.694828594729067,26.332810467437106,25.32357865070144,25.519725153530004,25.932925862058152,25.485766387351077,25.708025056063313,26.16690702017466,26.867124135227865,26.98936570844941,27.33184066115083,27.107100852081217,27.12869419143785,27.443845262815866,27.894633387840035,28.29203313076331,28.597604716754084,28.796096093353224,28.803382166262868,29.032765736099158,29.35788535033402,6.162072825832294,6.143508714049793,8.092695804302604,5.877885167788389,5.885393751961795,6.858922039672193,7.173628152093295,5.887861612013832,6.789788433998936,6.803124353051092,7.561838480181641,16.573013269385818,16.674043332632632,17.211154442311418,17.498440577833644,18.406469386475905,18.4564155007033,18.88282172500373,18.826967331421276,18.585603616784983,18.363118689020755,16.49961780910525,16.379566002722736,16.539949519163745,15.403941941468162,15.07201547062653,14.766970417969684,14.470728713478863,14.073725752935294,14.752648165951198,15.357957061499352,4.9203301275215505,4.889592288351954,5.753141702765589,4.221807053665121,4.230764988197734,4.342745039102928,4.9583035743370845,5.314852436146908,6.2737984707191705,6.186599003403452,6.378700823133462,1.4008132788450085,2.436978572525182,2.490969408279028,1.9608465229101066,1.7195843365208536,1.8793944258463307,1.243984412813758,0.9656723631243508,0.32198177025075675,-0.016115604037152523,16.46176933769758,16.1198772079529,15.937888698638048,15.74957451658323,15.287130272124791,15.049990815844788,14.948670980287597,16.560455527143418,16.71475322757477,15.818406860040968,8.373946005185969,8.333764904530074,8.384153532605264,8.807016548102865,8.451035640034789,7.9629921454751,7.705374327003133,6.692857619633551,5.744136157031675,5.598381320191161,23.220365029472052,23.157069111890618,23.08901685290636,21.176520545401303,20.976262558910058,20.767022952327068,20.4412841028258,20.286335743470378,20.350242193760817,20.2142540887746,-0.3409510141183396,0.07443494910842872,-0.08991327815196566,-1.1158202542491253,-1.4525155973148314,-1.1341996333339956,-1.1072608033756743,-1.1796289592895441,-0.05564306553230889,-0.02623083345675155,-3.805881622434131,-7.350782546266567,-6.143878107541909,-6.1569800198293665,-6.206144481530275,-7.392390097225909,-7.645797266305774,-4.6765795132258585,-4.648960018633978,-4.029431102830687,-3.9586140986215588,-11.106706889636358,-11.159671450055658,-11.110072205386455,-10.494818947853645,-10.445139721499507,-3.73016504176421,-3.812289169440395,-3.766752280722935,-10.204009980347115,-9.97828064213351,-10.01236176615111,-5.9899472271963266,-6.069828980523945,-6.325101258307426,-6.385419121948329,-7.901797285021365,-7.883252582280364,-8.139458600580292,-5.369977542829142,-5.500651839453681,-5.8791309465376855,-6.514291045422418,-6.8383917878643885,-8.85272255694874,-8.73214048195978,11.736104838321653,11.595548532680269,10.421554179793464,10.458406968606356,10.820555006755717,12.536617680955935,12.559611895422162,10.982289295049132,13.140608262435768,12.995969870759396,12.982365356109785,11.08919419042178,10.735929247542247,11.210612984836533,11.58593764174784,11.597387917417294,11.779661741545377],"y":[-0.6554589213947022,-0.42716947637072056,-0.04298901721500249,0.10794713393029651,0.2068934282835693,0.14486037121381973,-0.05575649849865895,-0.8548563957939784,-0.7535422847432809,-0.7398409421213271,-0.2727013306747742,10.898138197907818,10.920002189199762,11.698321477898913,11.270523268414522,10.348588001468375,10.315591255351466,10.68849256030036,11.002321135274707,11.149446044223222,11.339447445680687,11.439969940624977,14.65324852076776,14.794207778739498,15.244334264810862,15.91048999826189,16.254380721071623,16.772167969728546,16.984883661498387,16.012271809529896,15.939619745409022,13.088152023096711,14.784176096923467,15.107575470042352,15.373955447650006,15.303254921154645,14.598435025381946,14.154238627864679,14.219225084437587,14.260296679188416,14.387882511211675,13.27427609063873,13.612544747948245,15.883699315179486,15.512917404299914,14.268079763715166,14.668211989558385,13.877233545987488,13.726697607149816,14.609095368641356,15.80423634403593,15.37272505295515,15.755674346218498,4.171665071396848,4.358355461279157,1.9492355500620169,4.381272900806262,4.230337457287696,4.559541025229177,4.392282609870378,3.7851125602731646,3.1565559732776918,2.938868741354638,2.498042736345038,1.4221632656147642,1.1526041431409617,-0.6405030068437995,1.003279665667704,0.43071011811295024,0.1583540695039142,0.0426967763155139,-0.25930841183321524,-0.24804979027378815,-0.5707541467740097,-0.4274308491090652,20.442746307969415,20.796044365604764,20.85769527902412,22.14060177144457,22.217651084523744,22.331690526710247,23.117727512958457,22.51993162999931,22.785898720066765,21.02494946308062,-9.807717540482141,-9.737830455971794,-8.731038266175624,-8.773807049700704,-8.386201337642557,-8.465396603669342,-8.52805836605053,-8.821973146563584,-9.115096095336904,-9.00635271887266,-4.551613316233609,-4.193574826389333,-4.051846215537948,-3.574410624594552,-3.1954993320176217,-2.9154319975800247,-2.7229554746348414,-2.960519182373967,-3.1397434465182856,-3.5882545649882363,0.027411502803862342,0.14672134435117087,-0.26187553478278686,-0.45674101630647895,0.39432185267888104,-0.13021791906580726,-0.4509346815197544,-0.7675151634953391,-0.8068671731710403,-1.3856171169962952,18.26831368197393,18.33555455886255,19.54172730250134,18.836532214966002,18.84344129112668,19.061065703049984,19.834113648596215,19.82065284884012,20.18027649935189,20.279348975544906,19.718558751471644,-4.429555891725671,-4.4054607317130206,-4.375485010908324,-4.295343660045125,-4.473453014357826,-4.489925395766908,-4.790613733473582,-5.017543820180364,-5.494918428658252,-5.501039642367636,8.830755920420877,8.943029813213682,9.0447004979641,8.922306904475885,8.961475037241733,8.804232772459072,9.17316363750873,9.251844525283952,9.9144909255112,10.035730435952107,6.586564382427908,6.598939561745835,6.890493901775637,5.934225237529753,5.748439933306147,5.457221963249855,5.539307899638601,5.598615748460305,5.766917885484591,6.121817997759456,6.6553986086078165,17.622603694284262,17.65228273928218,17.950320190836628,18.526268492357897,18.85873276769844,19.09924979395788,19.287316995558246,19.319829025144752,19.44538333842069,19.515035026247073,16.37445294370312,16.968274873100857,17.152264306828283,17.79843492219453,18.29026064109368,18.998145503015675,18.95358224212231,19.36776894735876,19.499389429976212,19.3467049043652,22.916604658368993,23.304870951918936,23.51215029138869,24.811756842278562,24.549586978949694,24.72999816053834,24.849624728651303,24.30370143568144,23.982633584759107,23.846563912247415,7.1214880845660105,7.100284329254676,7.018797882233198,6.587582040886791,6.517727395461282,6.0486431637881815,6.612985692348345,6.70421849553062,6.977651189067914,7.453338707128197,22.080611701781514,22.94976094477385,22.914835587210302,23.119220690944147,22.9574741382941,22.70475237079312,23.14249846083837,24.015682293329508,24.429958241339754,24.40389850496782,-17.542466722535995,-14.685214569130093,-15.69968314505348,-15.859755215274658,-15.927612727700668,-15.476751824347296,-15.764082599349864,-14.864186894201728,-14.890024515326646,-13.829265926602385,-13.657894924757294,-14.478629231521357,-14.419862638494253,-14.47291706134391,-17.950455433333673,-17.941277102900973,-14.74470684901657,-16.26749181358657,-16.15352375963752,-16.297046865563534,-15.340013571391982,-15.916852900591616,-13.2765319573521,-13.362828393695034,-18.538278378955646,-18.621284336547816,-19.26367049292339,-19.19579107868821,-14.495367475630736,-16.743520993256162,-16.69858802969112,-14.351633915776592,-14.958981785933991,-17.457696279464,-17.348585580150743,-17.380570491330463,-15.12182172701353,-18.416097960020835,-18.597231756936374,-18.338492096010977,-17.929332031237315,-16.05837359429796,-15.609395424554096,-16.673733710093117,-15.32481400952367,-14.846141057412225,-16.816888390323584,-17.664894362541077,-19.347779577659086,-18.57733797921733,-17.70237498100047,-16.849987844673546,-18.928356384999248]}},"id":"6e210f27-9515-439e-886b-2ed95029ed12","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"f078f1e9-841e-431e-9db8-df298b13f107","type":"ColumnDataSource"},"glyph":{"id":"41eb632a-f3d0-47e8-a481-501b77d8be1b","type":"Circle"},"hover_glyph":null,"nonselection_glyph":{"id":"6811b0b7-f87f-436b-9e02-7f694e87a62f","type":"Circle"},"selection_glyph":null},"id":"ca17c5a0-7ac5-4ad3-b9cd-2279e6f39d73","type":"GlyphRenderer"},{"attributes":{},"id":"14a805c0-0707-44f3-82b7-49deb04fc6e6","type":"BasicTicker"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":20},"x":{"field":"x"},"y":{"field":"y"}},"id":"d8c413d8-f306-4561-84cf-54b896aead82","type":"Circle"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":20},"x":{"field":"x"},"y":{"field":"y"}},"id":"6811b0b7-f87f-436b-9e02-7f694e87a62f","type":"Circle"},{"attributes":{"fill_alpha":{"value":0.5},"fill_color":{"value":"blue"},"line_alpha":{"value":0.5},"line_color":{"value":"blue"},"size":{"units":"screen","value":20},"x":{"field":"x"},"y":{"field":"y"}},"id":"527f2f46-0c79-4069-a079-acb2a68bf596","type":"Circle"},{"attributes":{"data_source":{"id":"6e210f27-9515-439e-886b-2ed95029ed12","type":"ColumnDataSource"},"glyph":{"id":"527f2f46-0c79-4069-a079-acb2a68bf596","type":"Circle"},"hover_glyph":null,"nonselection_glyph":{"id":"d8c413d8-f306-4561-84cf-54b896aead82","type":"Circle"},"selection_glyph":null},"id":"071b7731-26f6-4e89-b314-52443f6b23a4","type":"GlyphRenderer"},{"attributes":{"callback":null},"id":"f81b6ace-d275-4e31-bac9-a53939b6f33b","type":"DataRange1d"},{"attributes":{"formatter":{"id":"06535a33-6b00-47df-8559-d96feff9cf07","type":"BasicTickFormatter"},"plot":{"id":"e0df0f74-b5b4-49d0-9f85-67dbfc01340d","subtype":"Figure","type":"Plot"},"ticker":{"id":"f6966f63-fff6-4fb8-90e7-0e26e65f3648","type":"BasicTicker"}},"id":"94140273-c3ea-4520-af88-bb79b7ece748","type":"LinearAxis"},{"attributes":{},"id":"f6966f63-fff6-4fb8-90e7-0e26e65f3648","type":"BasicTicker"},{"attributes":{"callback":null},"id":"c05f1234-445f-4e88-be31-ee00e3d44ac9","type":"DataRange1d"},{"attributes":{"active_drag":"auto","active_scroll":"auto","active_tap":"auto","tools":[{"id":"c7a7d445-3337-4387-9f83-d67c72c135ae","type":"HoverTool"}]},"id":"1c57f5d7-7eb6-4a37-a3e2-a9fe15bf78c7","type":"Toolbar"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":20},"x":{"field":"x"},"y":{"field":"y"}},"id":"fb30b46d-06a5-493e-9d2c-f99b5080a825","type":"Circle"},{"attributes":{"plot":{"id":"e0df0f74-b5b4-49d0-9f85-67dbfc01340d","subtype":"Figure","type":"Plot"},"ticker":{"id":"14a805c0-0707-44f3-82b7-49deb04fc6e6","type":"BasicTicker"}},"id":"8ce8ec66-00ca-472c-bcb3-7866589a49fa","type":"Grid"},{"attributes":{"below":[{"id":"3473d385-9348-47d5-bf4c-f690685ec729","type":"LinearAxis"}],"left":[{"id":"94140273-c3ea-4520-af88-bb79b7ece748","type":"LinearAxis"}],"plot_height":800,"plot_width":800,"renderers":[{"id":"3473d385-9348-47d5-bf4c-f690685ec729","type":"LinearAxis"},{"id":"8ce8ec66-00ca-472c-bcb3-7866589a49fa","type":"Grid"},{"id":"94140273-c3ea-4520-af88-bb79b7ece748","type":"LinearAxis"},{"id":"92000507-e361-474a-bc9e-52ea10fc8f86","type":"Grid"},{"id":"ca17c5a0-7ac5-4ad3-b9cd-2279e6f39d73","type":"GlyphRenderer"},{"id":"6d57e398-bd4f-40f4-99ae-1405e570093f","type":"GlyphRenderer"},{"id":"071b7731-26f6-4e89-b314-52443f6b23a4","type":"GlyphRenderer"},{"id":"0fa7e578-2a80-48f6-ad01-45b1d0aec379","type":"GlyphRenderer"}],"title":{"id":"a2d9a937-338d-41d0-8e0b-23f3bcc1e493","type":"Title"},"tool_events":{"id":"cf29c302-76bc-4fb2-aa91-91b48c2117b3","type":"ToolEvents"},"toolbar":{"id":"1c57f5d7-7eb6-4a37-a3e2-a9fe15bf78c7","type":"Toolbar"},"x_range":{"id":"c05f1234-445f-4e88-be31-ee00e3d44ac9","type":"DataRange1d"},"y_range":{"id":"f81b6ace-d275-4e31-bac9-a53939b6f33b","type":"DataRange1d"}},"id":"e0df0f74-b5b4-49d0-9f85-67dbfc01340d","subtype":"Figure","type":"Plot"}],"root_ids":["e0df0f74-b5b4-49d0-9f85-67dbfc01340d"]},"title":"Bokeh Application","version":"0.12.3"}};
                var render_items = [{"docid":"bf260907-8aea-413f-ad2a-41badba0755c","elementid":"bbbf02e2-2946-41b4-8bb1-e1cd8277265a","modelid":"e0df0f74-b5b4-49d0-9f85-67dbfc01340d"}];
                
                Bokeh.embed.embed_items(docs_json, render_items);
            });
          },
          function(Bokeh) {
          }
        ];
      
        function run_inline_js() {
          
          if ((window.Bokeh !== undefined) || (force === "1")) {
            for (var i = 0; i < inline_js.length; i++) {
              inline_js[i](window.Bokeh);
            }if (force === "1") {
              display_loaded();
            }} else if (Date.now() < window._bokeh_timeout) {
            setTimeout(run_inline_js, 100);
          } else if (!window._bokeh_failed_load) {
            console.log("Bokeh: BokehJS failed to load within specified timeout.");
            window._bokeh_failed_load = true;
          } else if (!force) {
            var cell = $("#bbbf02e2-2946-41b4-8bb1-e1cd8277265a").parents('.cell').data().cell;
            cell.output_area.append_execute_result(NB_LOAD_WARNING)
          }
      
        }
      
        if (window._bokeh_is_loading === 0) {
          console.log("Bokeh: BokehJS loaded, going straight to plotting");
          run_inline_js();
        } else {
          load_libs(js_urls, function() {
            console.log("Bokeh: BokehJS plotting callback run at", now());
            run_inline_js();
          });
        }
      }(this));
    </script>


利用SVM以轉化後的bottleneck features建立分類模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython2

    train_svm_classifier?

.. code:: ipython2

    train_svm_classifier(features, labels, sns, './model', split=True)


.. parsed-literal::

    train test split == True
    Fitting 3 folds for each of 20 candidates, totalling 60 fits
    [CV] kernel=linear, C=1 ..............................................
    [CV] ............... kernel=linear, C=1, score=0.982301, total=   0.3s
    [CV] kernel=linear, C=1 ..............................................


.. parsed-literal::

    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.3s remaining:    0.0s


.. parsed-literal::

    [CV] ............... kernel=linear, C=1, score=0.973214, total=   0.2s
    [CV] kernel=linear, C=1 ..............................................


.. parsed-literal::

    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.6s remaining:    0.0s


.. parsed-literal::

    [CV] ............... kernel=linear, C=1, score=0.963964, total=   0.2s
    [CV] kernel=linear, C=10 .............................................
    [CV] .............. kernel=linear, C=10, score=0.982301, total=   0.2s
    [CV] kernel=linear, C=10 .............................................
    [CV] .............. kernel=linear, C=10, score=0.973214, total=   0.2s
    [CV] kernel=linear, C=10 .............................................
    [CV] .............. kernel=linear, C=10, score=0.963964, total=   0.2s
    [CV] kernel=linear, C=100 ............................................
    [CV] ............. kernel=linear, C=100, score=0.982301, total=   0.2s
    [CV] kernel=linear, C=100 ............................................
    [CV] ............. kernel=linear, C=100, score=0.973214, total=   0.2s
    [CV] kernel=linear, C=100 ............................................
    [CV] ............. kernel=linear, C=100, score=0.963964, total=   0.2s
    [CV] kernel=linear, C=1000 ...........................................
    [CV] ............ kernel=linear, C=1000, score=0.982301, total=   0.3s
    [CV] kernel=linear, C=1000 ...........................................
    [CV] ............ kernel=linear, C=1000, score=0.973214, total=   0.3s
    [CV] kernel=linear, C=1000 ...........................................
    [CV] ............ kernel=linear, C=1000, score=0.963964, total=   0.2s
    [CV] kernel=rbf, C=1, gamma=0.01 .....................................
    [CV] ...... kernel=rbf, C=1, gamma=0.01, score=0.955752, total=   0.4s
    [CV] kernel=rbf, C=1, gamma=0.01 .....................................
    [CV] ...... kernel=rbf, C=1, gamma=0.01, score=0.973214, total=   0.4s
    [CV] kernel=rbf, C=1, gamma=0.01 .....................................
    [CV] ...... kernel=rbf, C=1, gamma=0.01, score=0.927928, total=   0.4s
    [CV] kernel=rbf, C=1, gamma=0.001 ....................................
    [CV] ..... kernel=rbf, C=1, gamma=0.001, score=0.566372, total=   0.4s
    [CV] kernel=rbf, C=1, gamma=0.001 ....................................
    [CV] ..... kernel=rbf, C=1, gamma=0.001, score=0.562500, total=   0.4s
    [CV] kernel=rbf, C=1, gamma=0.001 ....................................
    [CV] ..... kernel=rbf, C=1, gamma=0.001, score=0.567568, total=   0.4s
    [CV] kernel=rbf, C=1, gamma=0.0001 ...................................
    [CV] .... kernel=rbf, C=1, gamma=0.0001, score=0.566372, total=   0.4s
    [CV] kernel=rbf, C=1, gamma=0.0001 ...................................
    [CV] .... kernel=rbf, C=1, gamma=0.0001, score=0.562500, total=   0.4s
    [CV] kernel=rbf, C=1, gamma=0.0001 ...................................
    [CV] .... kernel=rbf, C=1, gamma=0.0001, score=0.567568, total=   0.4s
    [CV] kernel=rbf, C=1, gamma=1e-05 ....................................
    [CV] ..... kernel=rbf, C=1, gamma=1e-05, score=0.566372, total=   0.4s
    [CV] kernel=rbf, C=1, gamma=1e-05 ....................................
    [CV] ..... kernel=rbf, C=1, gamma=1e-05, score=0.562500, total=   0.4s
    [CV] kernel=rbf, C=1, gamma=1e-05 ....................................
    [CV] ..... kernel=rbf, C=1, gamma=1e-05, score=0.567568, total=   0.4s
    [CV] kernel=rbf, C=10, gamma=0.01 ....................................
    [CV] ..... kernel=rbf, C=10, gamma=0.01, score=0.973451, total=   0.3s
    [CV] kernel=rbf, C=10, gamma=0.01 ....................................
    [CV] ..... kernel=rbf, C=10, gamma=0.01, score=0.982143, total=   0.3s
    [CV] kernel=rbf, C=10, gamma=0.01 ....................................
    [CV] ..... kernel=rbf, C=10, gamma=0.01, score=0.963964, total=   0.3s
    [CV] kernel=rbf, C=10, gamma=0.001 ...................................
    [CV] .... kernel=rbf, C=10, gamma=0.001, score=0.955752, total=   0.4s
    [CV] kernel=rbf, C=10, gamma=0.001 ...................................
    [CV] .... kernel=rbf, C=10, gamma=0.001, score=0.973214, total=   0.4s
    [CV] kernel=rbf, C=10, gamma=0.001 ...................................
    [CV] .... kernel=rbf, C=10, gamma=0.001, score=0.927928, total=   0.4s
    [CV] kernel=rbf, C=10, gamma=0.0001 ..................................
    [CV] ... kernel=rbf, C=10, gamma=0.0001, score=0.566372, total=   0.4s
    [CV] kernel=rbf, C=10, gamma=0.0001 ..................................
    [CV] ... kernel=rbf, C=10, gamma=0.0001, score=0.562500, total=   0.4s
    [CV] kernel=rbf, C=10, gamma=0.0001 ..................................
    [CV] ... kernel=rbf, C=10, gamma=0.0001, score=0.567568, total=   0.4s
    [CV] kernel=rbf, C=10, gamma=1e-05 ...................................
    [CV] .... kernel=rbf, C=10, gamma=1e-05, score=0.566372, total=   0.4s
    [CV] kernel=rbf, C=10, gamma=1e-05 ...................................
    [CV] .... kernel=rbf, C=10, gamma=1e-05, score=0.562500, total=   0.4s
    [CV] kernel=rbf, C=10, gamma=1e-05 ...................................
    [CV] .... kernel=rbf, C=10, gamma=1e-05, score=0.567568, total=   0.4s
    [CV] kernel=rbf, C=100, gamma=0.01 ...................................
    [CV] .... kernel=rbf, C=100, gamma=0.01, score=0.991150, total=   0.3s
    [CV] kernel=rbf, C=100, gamma=0.01 ...................................
    [CV] .... kernel=rbf, C=100, gamma=0.01, score=0.973214, total=   0.3s
    [CV] kernel=rbf, C=100, gamma=0.01 ...................................
    [CV] .... kernel=rbf, C=100, gamma=0.01, score=0.963964, total=   0.3s
    [CV] kernel=rbf, C=100, gamma=0.001 ..................................
    [CV] ... kernel=rbf, C=100, gamma=0.001, score=0.973451, total=   0.3s
    [CV] kernel=rbf, C=100, gamma=0.001 ..................................
    [CV] ... kernel=rbf, C=100, gamma=0.001, score=0.982143, total=   0.3s
    [CV] kernel=rbf, C=100, gamma=0.001 ..................................
    [CV] ... kernel=rbf, C=100, gamma=0.001, score=0.963964, total=   0.3s
    [CV] kernel=rbf, C=100, gamma=0.0001 .................................
    [CV] .. kernel=rbf, C=100, gamma=0.0001, score=0.955752, total=   0.4s
    [CV] kernel=rbf, C=100, gamma=0.0001 .................................
    [CV] .. kernel=rbf, C=100, gamma=0.0001, score=0.973214, total=   0.4s
    [CV] kernel=rbf, C=100, gamma=0.0001 .................................
    [CV] .. kernel=rbf, C=100, gamma=0.0001, score=0.927928, total=   0.4s
    [CV] kernel=rbf, C=100, gamma=1e-05 ..................................
    [CV] ... kernel=rbf, C=100, gamma=1e-05, score=0.566372, total=   0.4s
    [CV] kernel=rbf, C=100, gamma=1e-05 ..................................
    [CV] ... kernel=rbf, C=100, gamma=1e-05, score=0.562500, total=   0.4s
    [CV] kernel=rbf, C=100, gamma=1e-05 ..................................
    [CV] ... kernel=rbf, C=100, gamma=1e-05, score=0.567568, total=   0.4s
    [CV] kernel=rbf, C=1000, gamma=0.01 ..................................
    [CV] ... kernel=rbf, C=1000, gamma=0.01, score=0.991150, total=   0.3s
    [CV] kernel=rbf, C=1000, gamma=0.01 ..................................
    [CV] ... kernel=rbf, C=1000, gamma=0.01, score=0.973214, total=   0.3s
    [CV] kernel=rbf, C=1000, gamma=0.01 ..................................
    [CV] ... kernel=rbf, C=1000, gamma=0.01, score=0.963964, total=   0.3s
    [CV] kernel=rbf, C=1000, gamma=0.001 .................................
    [CV] .. kernel=rbf, C=1000, gamma=0.001, score=0.991150, total=   0.3s
    [CV] kernel=rbf, C=1000, gamma=0.001 .................................
    [CV] .. kernel=rbf, C=1000, gamma=0.001, score=0.973214, total=   0.3s
    [CV] kernel=rbf, C=1000, gamma=0.001 .................................
    [CV] .. kernel=rbf, C=1000, gamma=0.001, score=0.963964, total=   0.3s
    [CV] kernel=rbf, C=1000, gamma=0.0001 ................................
    [CV] . kernel=rbf, C=1000, gamma=0.0001, score=0.973451, total=   0.3s
    [CV] kernel=rbf, C=1000, gamma=0.0001 ................................
    [CV] . kernel=rbf, C=1000, gamma=0.0001, score=0.982143, total=   0.3s
    [CV] kernel=rbf, C=1000, gamma=0.0001 ................................
    [CV] . kernel=rbf, C=1000, gamma=0.0001, score=0.963964, total=   0.3s
    [CV] kernel=rbf, C=1000, gamma=1e-05 .................................
    [CV] .. kernel=rbf, C=1000, gamma=1e-05, score=0.955752, total=   0.4s
    [CV] kernel=rbf, C=1000, gamma=1e-05 .................................
    [CV] .. kernel=rbf, C=1000, gamma=1e-05, score=0.973214, total=   0.4s
    [CV] kernel=rbf, C=1000, gamma=1e-05 .................................
    [CV] .. kernel=rbf, C=1000, gamma=1e-05, score=0.927928, total=   0.4s


.. parsed-literal::

    [Parallel(n_jobs=1)]: Done  60 out of  60 | elapsed:   23.2s finished


.. parsed-literal::

    
    Best parameters set:
    SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma=0.01, kernel='rbf',
      max_iter=-1, probability=True, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
    
    Confusion matrix:
    Labels: 0,1,4,7
    
    [[13  0  0  0]
     [ 0 14  0  0]
     [ 0  0 70  0]
     [ 0  1  0 14]]
    
    Classification report:
                 precision    recall  f1-score   support
    
              0       1.00      1.00      1.00        13
              1       0.93      1.00      0.97        14
              4       1.00      1.00      1.00        70
              7       1.00      0.93      0.97        15
    
    avg / total       0.99      0.99      0.99       112
    




.. parsed-literal::

    GridSearchCV(cv=3, error_score='raise',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=True, random_state=None, shrinking=True,
      tol=0.001, verbose=False),
           fit_params={}, iid=True, n_jobs=1,
           param_grid=[{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}, {'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001, 1e-05]}],
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring=None, verbose=3)



