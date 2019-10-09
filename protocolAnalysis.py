######## MODULE FOR ANALYZING LOGS DATA FROM STIMULATION PROTOCOLS

'''
def video_from_frames():

    writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30,(640,480))
    for frame in range(1000):
        writer.write(np.random.randint(0, 255, (480,640,3)).astype('uint8'))
    writer.release()
'''
import numpy as np
import matplotlib.pyplot as plt


def analyze_log_threshold_NEW(f,name,show_figures):
    
    display(f)
    electrodeList = []
    thresholdList = []
    threholdList_temp = []

    GroupOrSingle = 'Single'
    SummaryFlag = False
    Group_FLAG = False

    old_E = 0
    remap = 0

    first_electrode_FLAG = 1

    a = name.split('#')
    a = a[0]
    a = a[-10:-5]
    Day_Month = a 
    
    lineCounter = 0
    
    for line in f:   
        
        if lineCounter == 2:
            print('line ' + line[0:48])
            standard_config = '[300; 166.66667; 1; 1; 1; 170; 170; 2933; 60; 1]'
            if line[0:48] != standard_config:
                print('Standard config ',standard_config)
                print('Current config  ',line[0:48])
                print('NON STANDARD CONFIGURATION: SKIPPING LOG. CHECK SOURCE CODE FOR CONFIG' )
                break
            
        #Parsing only the "Summary values"
        if SummaryFlag == True:
            
            if line[0:5] == 'Group': #This means we are working with electrode groups
                Group_FLAG = True
                GroupOrSingle = 'Group'
                

                first_split = line.split()
                E = first_split[1]
                E = E[2:-1].split('_')
                I = int(first_split[2])
                
                for k in range(len(E)):
                    electrodeList.append(int(E[k]))
                    thresholdList.append(int(I))
                 
            if line[0] == 'E': #This means we are working with single electrodes
                first_split = line.split()
                second_split = first_split[0:-1]

                E = second_split[0]
                E = int(E[1:-1])
                I = int(second_split[1])
                
                electrodeList.append(int(E))
                thresholdList.append(int(I))
                
        if(line[0:8] == 'Summary:'):
            SummaryFlag = True
        lineCounter = lineCounter + 1
        

    thresholdList = np.array(thresholdList)
    electrodeList = np.squeeze(electrodeList)
    
    SummaryFlag = False  

    electrodeList_completed = np.zeros((96))
    thresholdList_completed = np.zeros((96))
    #print(' hola ', electrodeList.shape)
    
    electrodeList_completed[electrodeList-1] = electrodeList
    thresholdList_completed[electrodeList-1] = thresholdList
    
    if show_figures == True:
    
        fig_2= plt.figure(figsize = (20,3))
        plt.plot(np.arange(1, 97),thresholdList_completed, color = "blue", linewidth = 2, marker = 'o')
        plt.title('thresholdList after ordering ' + name)
        plt.ylabel('Intensity in uA')
        plt.xlabel('Electrode number')
        plt.xticks(np.arange(1, 96, 2))
    
    a = array96ToMatrix(thresholdList_completed)

    if show_figures == True:
        
        fig_3= plt.figure(figsize=(5,5))
        plt.imshow(a,cmap='hot_r',vmin=0, vmax=120)
        plt.title('thresholds after ordering ' + Day_Month)
        plt.colorbar()

    return GroupOrSingle, Day_Month, electrodeList_completed, thresholdList_completed, a


def array96ToMatrix(array):

    value = 0
    
    index = 0
    a = np.insert(array[:],index,value)
    index = 9
    a = np.insert(a,index,value)
    index = 90
    a = np.insert(a,index,value)
    index = 99
    a = np.insert(a,index,value)
    a = a.reshape(10,10)
    
    a = np.fliplr(a)

    return a

def ChannelToElectrode_OLD(channels):
    '''
    The impedance indexes are CHANNELS. 
    This function returns the ELECTRODE that those channels activates
    
    Example: Impedance Channel 95 is 40KOhm
    This means that the Electrode 1's impedance is 40KOhm 
    
    Ejemplo:
    channels = np.array([95,32,65])
    electrodes = ChannelToElectrode(channels)
    electrodes = [1,2,96]
    '''

    channelMap = np.array([95,32,30,28,26,24,22,18,
    96,63,61,64,31,29,27,20,16,14,
    93,94,59,60,62,21,25,23,12,10,
    92,91,57,58,52,54,19,13,11,8,
    90,89,55,56,46,50,15,17,9,6,
    88,87,53,51,44,42,48,5,7,4,
    85,86,49,47,43,40,38,36,34,3,
    83,84,82,45,41,39,37,35,33,1,
    81,80,78,76,74,72,70,68,66,2,
    79,77,75,73,71,69,67,65])

    electrodeList = []

    for i in range(channels.shape[0]):

        a = np.where(channelMap == channels[i])
        a = np.array(a).astype('uint8')
        a = a + 1
        electrodeList.append(a)

    electrodes = np.array(electrodeList).astype('uint8') 
    
    return np.transpose(electrodes)

def ChannelToElectrode(channels,thresholds=None):
    '''
    The impedance indexes are CHANNELS. 
    This function returns the ELECTRODE that those channels activates
    
    If thresholds for each channel are introduced, it also reorders those thresholds
    
    Example: Impedance Channel 95 is 40KOhm
    This means that the Electrode 1's threshold is 40KOhm 
    
    Ejemplo:
    channels = np.array([95,32,65])
    electrodes = ChannelToElectrode(channels)
    electrodes = [1,2,96]
    '''

    channelMap = np.array([95,32,30,28,26,24,22,18,
    96,63,61,64,31,29,27,20,16,14,
    93,94,59,60,62,21,25,23,12,10,
    92,91,57,58,52,54,19,13,11,8,
    90,89,55,56,46,50,15,17,9,6,
    88,87,53,51,44,42,48,5,7,4,
    85,86,49,47,43,40,38,36,34,3,
    83,84,82,45,41,39,37,35,33,1,
    81,80,78,76,74,72,70,68,66,2,
    79,77,75,73,71,69,67,65])

    electrodeList = []
    if thresholds.all() != None:
        thresholds_temp = []

    for i in range(channels.shape[0]):

        a = np.where(channelMap == channels[i])
        a = np.array(a).astype('uint8')
        a = a + 1           #to have an electrodeList starting from 1, not 0 
        electrodeList.append(a)
        
        #display(" en el canal " + str(i) + " hay un threshold de " +str(channels[i]) + " uA")
        #display(" en el electrodo " + str(a) + " hay un threshold de " +str(thresholds[a-1]) + " uA")
        
        if thresholds.all() != None:
            thresholds_temp.append(thresholds[a-1])
        
    if thresholds.all() != None:
        thresholds = thresholds_temp
        thresholds = np.array(thresholds).astype('uint8') 
    
    electrodes = np.array(electrodeList).astype('uint8') 
    if thresholds.all() != None:
        return np.transpose(electrodes),np.transpose(thresholds)
    else:
        return np.transpose(electrodes)
        

def ElectrodeToChannel(electrodes):

    '''
    Maps electrodes to channels
    
    Example:
    
    electrodes = np.array([1,2,96])
    channels = ElectrodeToChannel(electrodes)
    channels = [95,32,53]
    
    '''

    channelMap = np.array([95,32,30,28,26,24,22,18,
    96,63,61,64,31,29,27,20,16,14,
    93,94,59,60,62,21,25,23,12,10,
    92,91,57,58,52,54,19,13,11,8,
    90,89,55,56,46,50,15,17,9,6,
    88,87,53,51,44,42,48,5,7,4,
    85,86,49,47,43,40,38,36,34,3,
    83,84,82,45,41,39,37,35,33,1,
    81,80,78,76,74,72,70,68,66,2,
    79,77,75,73,71,69,67,65])

    channelList = []

    for i in range(electrodes.shape[0]):

        channelList.append(channelMap[electrodes[i]-1])

    channels = np.array(channelList).astype('uint8')     
    return np.transpose(channels)
    

def ButtonIndexToElectrode(vector):

    '''
    Fixes a bug on our button naming

    '''

    channelMap = np.array([95,32,30,28,26,24,22,18,
    96,63,61,64,31,29,27,20,16,14,
    93,94,59,60,62,21,25,23,12,10,
    92,91,57,58,52,54,19,13,11,8,
    90,89,55,56,46,50,15,17,9,6,
    88,87,53,51,44,42,48,5,7,4,
    85,86,49,47,43,40,38,36,34,3,
    83,84,82,45,41,39,37,35,33,1,
    81,80,78,76,74,72,70,68,66,2,
    79,77,75,73,71,69,67,65])


    channelMap_Inverted = np.array([
        18, 22, 24, 26, 28, 30, 32, 95,
       14, 16, 20, 27, 29, 31, 64, 61, 63, 96,
       10, 12, 23, 25, 21, 62, 60, 59, 94, 93,
        8, 11, 13, 19, 54, 52, 58, 57, 91, 92,
        6,  9, 17, 15, 50, 46, 56, 55, 89, 90,
        4,  7,  5, 48, 42, 44, 51, 53, 87, 88,
        3, 34, 36, 38, 40, 43, 47, 49, 86, 85,
        1, 33, 35, 37, 39, 41, 45, 82, 84, 83,
        2, 66, 68, 70, 72, 74, 76, 78, 80, 81,
            65, 67, 69, 71, 73, 75, 77, 79
    ])

    resultList = []

    for i in range(vector.shape[0]):

        index = np.where(channelMap_Inverted == vector[i])
        a = channelMap[index]
        resultList.append(a)
    result = np.array(resultList).astype('uint8')  

    return np.transpose(result)

def show_text(f):
    
    '''
    Shows the complete text line by line
    '''
    
    for line in f:
        print(line)
        
def analyze_log_threshold(f,name):
    " "
    
    display(f)
    electrodeList = []
    thresholdList = []
    threholdList_temp = []

    SummaryFlag = False

    old_E = 0
    remap = 0

    first_electrode_FLAG = 1
    
    for line in f:

        #Parsing only the "Summary values"
        if SummaryFlag == True:
            if line[0] == 'E':

                first_split = line.split()
                second_split = first_split[0:-1]

                E = second_split[0]
                E = int(E[1:-1])
                I = int(second_split[1])

                #print("E " + str(E))
                #print("old_E " + str(old_E))
                #print("I " + str(I))
                #print(" ")

                ################## NEW CODE 
                if first_electrode_FLAG == 1:
                    if (int(E) != 1):
                        print("Inserting missing Electrode " + str(1) + " with threshold = 0")
                        
                        electrodeList.append(int(1))
                        thresholdList.append(0)
                        old_E += 1

                        first_electrode_FLAG == 0
                ################ END NEW CODE

                if (int(E) != int(old_E)+1):
                    print("Inserting missing Electrode " + str(old_E+1) + " with threshold = 0")
                    
                    if old_E+1 == 3: #if the element missing is a three, we were storing channels in that log instead of electrodes
                        remap = 1
                        
                    electrodeList.append(int(old_E+1))
                    thresholdList.append(0)
                    old_E += 1

                electrodeList.append(int(E))
                thresholdList.append(int(I))
                
                old_E += 1

        if(line[0:8] == 'Summary:'):
            SummaryFlag = True


    SummaryFlag = False
    
    #the thresholds are computed for all the electrodes but the electrode 3, since it has "infinite" impedance.
    #Therefore, we insert a "0" for that electrode threshold and we are ready to represent the values

    
    if remap == 1:
        electrodeList,thresholdList_BLA_NOTUSED = ChannelToElectrode(np.array(electrodeList),np.array(thresholdList))
        #OVERRIDING THRESHOLDLIST
        electrodeList = np.squeeze(electrodeList[0,0,:])
        thresholdList_temp = []

    thresholdList = np.array(thresholdList)
    thresholdList = thresholdList[np.array(np.squeeze(electrodeList)-1).astype('uint8')]
    electrodeList = np.squeeze(electrodeList)
        
    #display("electrodeList " + str(np.squeeze(electrodeList)))
    #display("thresholdList " + str(np.squeeze(thresholdList)))
   
    thresholdList_completed = np.squeeze(thresholdList)
    electrodeList_completed = electrodeList
    
    fig_2= plt.figure(figsize = (20,3))
    plt.plot(np.arange(1, 97),thresholdList_completed, color = "blue", linewidth = 2, marker = 'o')
    plt.title('thresholdList after ordering ' + name)
    plt.ylabel('Intensity in uA')
    plt.xlabel('Electrode number')
    plt.xticks(np.arange(1, 96, 2))
    
    a = array96ToMatrix(thresholdList_completed)

    fig_3= plt.figure(figsize=(5,5))
    plt.imshow(a,cmap='hot_r')
    plt.title('thresholds after ordering')
    plt.colorbar()
    
    return electrodeList, thresholdList_completed, a

def analyze_Intensity_2AFC(f):
    '''
    '''
    electrodeList = []
    intensityList = []
    accuracyList = []
    thresholdList = []

    for line in f:

        if (line[0:10] == 'Electrode:'):
            electrode = line[11:-1]
            print("Electrode: " + str(electrode))

        elif (line[0:7] == 'Trials:'):
            trials = line[8:-1]
            print("Trials: " + str(trials))

        elif (line[0:16] == 'Parameter Order:'):   
            config = line[17:-1] + " "

        elif (line[0:13] == 'Configuration'): 
            config = config + line[44:-1]

            print("Configuration: " + str(config))

        elif (line[0:18] == 'Minimum intensity:'):
            l = line.split()
            min_intensity = l[2]
            print("min_intensity: " + min_intensity)

        elif (line[0:18] == 'Maximum intensity:'):
            l = line.split()
            max_intensity = l[2]
            print("max_intensity: " + max_intensity)

        elif (line[0:11] == 'Intensity :'):
            l = line.split()
            intensityList.append(l[2])

        elif (line[0:9] == 'Accuracy:'):
            l = line.split()
            l = str(l[1])
            l = l[0:-1]
            accuracyList.append(l)  

    intensityList = np.array(intensityList).astype('float64')
    accuracyList = np.array(accuracyList).astype('float64')

    print("Intensities: " + str(intensityList))
    print("Accuracies: " + str(accuracyList))

    fig = plt.figure(figsize = (15,5))
    plt.plot(intensityList,accuracyList, color = "blue", marker='o', linewidth = 2)
    plt.ylim((0, 100))
    plt.title('Accuracy(%) vs Intensity(uA), electrode ' + str(electrode))

    return electrode,trials,config,min_intensity,max_intensity,intensityList,accuracyList



def analyze_impedances(f):
    
    datesList = []
    channelList = []
    impedanceList = []

    lineCounter = 0

    for line in f:

        s = line.split()

        #Getting the dates in the first line
        if lineCounter == 0:
            datesList = s
            lineCounter = 1

        else:
            channelList.append(s[0])
            impedanceList.append(s[1:-1])

    num_channels = len(impedanceList)
    num_days = len(impedanceList[0])

    impedancesData = []

    for i in range(num_channels):

        channelData = impedanceList[i]
        #print(channelData)
        temp = []

        for i in range(num_days):
            
            value = channelData[i]
            commas_removed = value.replace(',', '.')
            temp.append(commas_removed)

        impedancesData.append(temp)
    
    
    impedancesArray = np.array(impedancesData).astype('float64')
    impedancesArray = np.transpose(impedancesArray)
    
    fig_2 = plt.figure(figsize = (15,5))
    #plt.plot(impedancesArray, marker='o', linewidth = 1)
    #plt.title('impedances through time')
    plt.semilogy(impedancesArray, marker='o', linewidth = 1)
    plt.title('impedances in logarithmic scale, through time')
    
    return datesList,impedancesArray


def analyze_OLD_psichophysic_fast(f):
    
    electrode = 0
    intensitiesList = []
    scoresList = []

    for line in f:
        

        if (line[0:10] == 'Electrode:'):

            electrode = line[11:-1]
         
        elif (line[0:16] == 'Parameter Order:'):   
            config = line[17:-1] + " "
                
        elif (line[0:44] == 'Configuration Intensity Psichometric Curves:'): 
            config = config + '\n' + line[44:-1]


        elif (line[0:18] == 'Stimulating with :'):

            intensitiesList.append(str(line[19:21]))

        if (line[0:19] == 'Phosphene perceived'):

            scoresList.append(1)

        if (line[0:23] == 'No phosphene perception'):

            scoresList.append(0)  

    intensitiesList = np.array(intensitiesList).astype('uint8')
    scoresList = np.array(scoresList).astype('uint8')
    
    

    #finding unique values
    uni = np.unique(intensitiesList)
    accumulated_scores = []

    for i in range(len(uni)):
        a = np.where(intensitiesList  == uni[i])
        suma = np.sum(scoresList[a])
        accumulated_scores.append(suma)
  
    num_trials = len(intensitiesList) / np.squeeze(np.array([len(uni)]).astype('uint8'))
    accumulated_scores = np.array(accumulated_scores).astype('uint8')
    percentage_scores = (accumulated_scores/float(num_trials)*100)
    
    print("electrode " + str(electrode))
    print("Total Number of Trials : " + str(len(intensitiesList)))
    print("Configuration: " + str(config))
    print("intensities  " + str(uni))
    print("acc scores   " + str(accumulated_scores))
    print("acc scores   " + str(percentage_scores))
    print(" ")

    fig = plt.figure(figsize = (15,5))
    plt.plot(uni,percentage_scores, color = "blue", marker='o', linewidth = 2)
    plt.ylim((0, 100))
    plt.title('Times perceived(%) vs Intensity(uA), electrode ' + str(electrode))

    return uni, percentage_scores

def analyze_psichophysic_fast(f,show_prints):

    electrodes = []
    intensitiesList = []
    scoresList = []
    timingsList = []

    Stimulus_type = '' #This will be either single train or blinking


    for line in f:


        if (line[0:19] == 'Configuration Tags:'):
            configuration_tags = line[20:-1]
            print('config tags ' + str(configuration_tags) + '\n') 

        elif (line[0:14] == 'Configuration:'):

            config = line[15:-1]
            print('config ' + str(config) + '\n') 

        elif (line[0:10] == 'Electrode:'):

            electrodes = line[11:-1]
            #electrodes = electrodes[1:-1]
            print('electrodes ' + str(electrodes) + '\n') 

        elif (line[0:15] == 'Single Stimulus'):

            Stimulus_type = 'Single Stimulus'
            print('Stimulus_type ' + Stimulus_type + '\n') 

        elif (line[0:8] == 'Blinking'):

            Stimulus_type = 'Blinking with repetitions=10 and train frequency=3Hz'
            print('Stimulus_type' + Stimulus_type + '\n') 

        elif (line[0:3] == 'Is:'):

            intensities = line[3:-1]

            a = intensities.split(",")

            for j in range(len(a)):

                if j == 0:
                    a_aux = a[j]
                    a_aux = a_aux.split('[')
                    a_aux = a_aux[1]
                    intensitiesList.append(a_aux)  

                elif j == len(a)-1:
                    a_aux = a[j]
                    a_aux = a_aux.split(']')
                    a_aux = a_aux[0]
                    intensitiesList.append(a_aux)  

                else:
                    intensitiesList.append(a[j])


        elif (line[0:3] == 'Rs:'):

            responses = line[3:-1]

            a = responses.split(",")

            for j in range(len(a)):

                if j == 0:
                    a_aux = a[j]
                    a_aux = a_aux.split('[')
                    a_aux = a_aux[1]
                    scoresList.append(a_aux)  

                elif j == len(a)-1:
                    a_aux = a[j]
                    a_aux = a_aux.split(']')
                    a_aux = a_aux[0]
                    scoresList.append(a_aux)  

                else:
                    scoresList.append(a[j])


        elif (line[0:3] == 'Ds:'):

            timings = line[3:-1]

            a = timings.split(",")
            #print("timings : " + str(a) +'\n') 

            for j in range(len(a)):

                if j == 0:
                    a_aux = a[j]
                    a_aux = a_aux.split('[')
                    a_aux = str(a_aux[1])
                    #print("a_aux : l" + str(a_aux) +'\n') 
                    timingsList.append(a_aux[1:-1])
                    
                elif j == len(a)-1:
                    a_aux = a[j]
                    a_aux = a_aux.split(']')
                    #print("a_aux : " + str(a_aux) +'\n') 
                    a_aux = a_aux[0:]
                    timingsList.append(a_aux)  

                else:
                    #print("a : l" + str(a[j]) +'\n') 
                    a_aux = str(a[j])
                    timingsList.append(a_aux[1:-1])    
                    #print("a : l" + str(a_aux[1:-1]) +'\n') 


    intensitiesList = np.array(intensitiesList).astype('uint8')
    scoresList = np.array(scoresList).astype('uint8')
    #timingsList = np.array(timingsList).astype('uint8')


    #finding unique values
    uni = np.unique(intensitiesList)
    accumulated_scores = []
    accumulated_timings = []

    for i in range(len(uni)):
        a = np.where(intensitiesList  == uni[i])

        #print("respuesta " + str(scoresList[a]))
        suma_scores = np.sum(scoresList[a])
        accumulated_scores.append(suma_scores)

        #suma_timings = np.sum(np.array(timingsList[a]))
        #accumulated_timings.append(suma_timings)
        #print("timing for  uni[i] " +str(uni[i]) + str(timingsList[a]) +'\n') 

    num_trials = len(intensitiesList) / np.squeeze(np.array([len(uni)]).astype('uint8'))

    accumulated_scores = np.array(accumulated_scores).astype('uint8')
    percentage_scores = (accumulated_scores/float(num_trials)*100)

    accumulated_timings = np.array(accumulated_timings).astype('uint8')
    mean_timings = (accumulated_timings/float(num_trials))
    
    if show_prints == 1:
        print("Total Number of Trials : " + str(len(intensitiesList)) +'\n') 
        print("Configuration: " + str(config)+'\n') 
        print("intensities  " + str(uni)+'\n') 
        print("acc scores   " + str(accumulated_scores) +'\n') 
        print("percentage scores   " + str(percentage_scores) +'\n') 
        #print("mean timings   " + str(mean_timings) +'\n') 
        print("num trials   " + str(num_trials) +'\n') 
        print(" ")

    plt.figure(figsize = (10,3))
    plt.plot(uni,percentage_scores, color = "blue", marker='o', linewidth = 2)
    plt.ylim((0, 100))
    plt.title('Perceptions(%) vs Intensity(uA), electrode ' + str(electrodes) + ' ' + str(config))


    return uni, percentage_scores

def analyze_psichophysic_fast_wTimeWindow(f,LOW_TIME_THRESHOLD,HIGH_TIME_THRESHOLD,show_prints):
    
    print(' show prints ', show_prints)
    electrodes = []
    intensitiesList = []
    scoresList = []
    timingsList = []
    num_valid_trials = []

    Stimulus_type = '' #This will be either single train or blinking
    
    experimentInfo =	{
      "config tags": '',
      "config": '',
      "electrodes": '',
      "Stimulus_type": '',
      "valid_trials_perIntensity": ''
    }

    for line in f:

        if (line[0:19] == 'Configuration Tags:'):
            configuration_tags = line[20:-1]
            #print('config tags ' + str(configuration_tags) + '\n') 
            experimentInfo["config tags"] = str(configuration_tags)

        elif (line[0:14] == 'Configuration:'):

            config = line[15:-1]
            #print('config ' + str(config) + '\n') 
            experimentInfo["config"] = str(config)

        elif (line[0:10] == 'Electrode:'):

            electrodes = line[11:-1]
            #electrodes = electrodes[1:-1]
            #print('electrode(s) ' + str(electrodes) + '\n') 
            experimentInfo["electrodes"] = str(electrodes)

        elif (line[0:15] == 'Single Stimulus'):

            Stimulus_type = 'Single Stimulus'
            #print('Stimulus_type ' + Stimulus_type + '\n') 
            experimentInfo["Stimulus_type"] = Stimulus_type

        elif (line[0:8] == 'Blinking'):
            #Stimulus_type = 'Blinking with repetitions=10 and train frequency=3Hz'
            Stimulus_type = 'Blinking with repetitions=10 and train frequency=3Hz'
            #print('Stimulus_type' + Stimulus_type + '\n') 
            experimentInfo["Stimulus_type"] = Stimulus_type

        elif (line[0:3] == 'Is:'):

            intensities = line[3:-1]

            a = intensities.split(",")

            for j in range(len(a)):

                if j == 0:
                    a_aux = a[j]
                    a_aux = a_aux.split('[')
                    a_aux = a_aux[1]
                    intensitiesList.append(a_aux)  

                elif j == len(a)-1:
                    a_aux = a[j]
                    a_aux = a_aux.split(']')
                    a_aux = a_aux[0]
                    intensitiesList.append(a_aux)  

                else:
                    intensitiesList.append(a[j])


        elif (line[0:3] == 'Rs:'):

            responses = line[3:-1]

            a = responses.split(",")

            for j in range(len(a)):

                if j == 0:
                    a_aux = a[j]
                    a_aux = a_aux.split('[')
                    a_aux = a_aux[1]
                    scoresList.append(a_aux)  

                elif j == len(a)-1:
                    a_aux = a[j]
                    a_aux = a_aux.split(']')
                    a_aux = a_aux[0]
                    scoresList.append(a_aux)  

                else:
                    scoresList.append(a[j])

        elif (line[0:3] == 'Ds:'):

            timings = line[3:-1]

            a = timings.split(",")
            #print("timings : " + str(a) +'\n') 

            for j in range(len(a)):

                if j == 0:
                    a_aux = a[j]
                    a_aux = a_aux.split('[')
                    a_aux = str(a_aux[1])
                    #print("a_aux : l" + str(a_aux) +'\n') 
                    #timingsList.append(a_aux[0:-1])
                    timingsList.append(a_aux)

                elif j == len(a)-1:
                    a_aux = a[j]
                    a_aux = a_aux.split(']')
                    a_aux = a_aux[0]
                    #print("a_aux : " + str(a_aux[1:len(a_aux)]) +'\n') 
                    timingsList.append(a_aux[0:len(a_aux)])  

                else:
                    #print("a : l" + str(a[j]) +'\n') 
                    a_aux = str(a[j])
                    #timingsList.append(a_aux[1:-1])    
                    timingsList.append(a_aux[1:len(a_aux)]) 
                    #print("a : l" + str(a_aux[1:-1]) +'\n') 


    intensitiesList = np.array(intensitiesList).astype('uint32')
    scoresList = np.array(scoresList).astype('uint32')
    timingsList = np.array(timingsList).astype('uint32')

    #finding unique values
    uni = np.unique(intensitiesList)
    accumulated_scores = []
    accumulated_timings = []

    #Obtaining all the intensities, scores and times
    for i in range(len(uni)):
        a = np.where(intensitiesList  == uni[i])

        #Checking if the timings are in the correct interval. If they are too slow or too quick, depending
        #on the time thresholds, those scores are not valid
        tim = timingsList[a]
        new_tim = tim
        new_scores = scoresList[a]
        
        del_counter = 0   
        for j in range(len(tim)):
            if tim[j] < LOW_TIME_THRESHOLD or tim[j] > HIGH_TIME_THRESHOLD:
                
                #print('Deleting timing out of threshold: ' + str(tim[j]))
                #print('Checking deleted timing: ' + str(new_tim[del_counter]))
                
                new_tim = np.delete(new_tim,del_counter)
                new_scores = np.delete(new_scores,del_counter)
                del_counter = del_counter - 1
                
            del_counter = del_counter + 1

        num_valid_trials.append(len(new_scores))
        suma_scores = np.sum(new_scores)
        suma_timings = np.sum(new_tim)

        accumulated_scores.append(suma_scores)
        accumulated_timings.append(suma_timings)

    num_valid_trials = np.array(num_valid_trials).astype('uint32')
    #print("Num_valid_trials per Intensity " + str(num_valid_trials))
    experimentInfo["valid_trials_perIntensity"] = num_valid_trials
    

    accumulated_scores = np.array(accumulated_scores).astype('uint32')
    percentage_scores = (accumulated_scores/num_valid_trials*100)

    accumulated_timings = np.array(accumulated_timings).astype('uint32')
    mean_timings = (accumulated_timings/num_valid_trials)
    print(' show prints ', show_prints)
    print(' show prints ', show_prints)
    print(' show prints ', show_prints)
    print(' show prints ', show_prints)
    print(' show prints ', show_prints)
    print(' show prints ', show_prints)
    print(' show prints ', show_prints)
    print(' show prints ', show_prints)
    print(' show prints ', show_prints)
    
    if show_prints == 1:
        
        print("Total Number of Trials : " + str(len(intensitiesList)) +'\n') 
        print("Configuration: " + str(config)+'\n') 
        print("intensities  " + str(uni)+'\n') 
        print("acc scores   " + str(accumulated_scores) +'\n') 
        print("percentage scores   " + str(percentage_scores) +'\n') 
        print("mean timings   " + str(mean_timings) +'\n') 
        print("num trials   " + str(num_valid_trials) +'\n') 
        print(" ")

        plt.figure(figsize = (10,3))
        plt.plot(uni,percentage_scores, color = "blue", marker='o', linewidth = 2)
        plt.ylim((0, 100))
        plt.title('Perceptions(%) vs Intensity(uA), electrode ' + str(electrodes) + ' ' + str(config))
    print(' ********************* ' )
    print(' ********************* ' )
    print(' ********************* ' )
    print(' ********************* ' )
    print(' ********************* ' )
    print(' ********************* ' )
    print(' EXPERIMENT INFO ', experimentInfo)
    print(' ********************* ' )
    print(' ********************* ' )
    print(' ********************* ' )
    print(' ********************* ' )
    
    return np.squeeze(uni), np.squeeze(percentage_scores), experimentInfo

def analyze_Intensity_2AFC_oneIntensity_new(f):
    '''
    '''
    electrodeList = []
    intensityList = []
    thresholdList = []
    
    correctList = []
    patientList = []

    for line in f:

        if (line[0:10] == 'Electrode:'):
            electrode = line[11:-1]
            print("Electrode: " + str(electrode))

        elif (line[0:7] == 'Trials:'):
            trials = line[8:-1]
            print("Trials: " + str(trials))

        elif (line[0:16] == 'Parameter Order:'):   
            config = line[17:-1] + " "

        elif (line[0:13] == 'Configuration'): 
            config = config + line[44:-1]

            print("Configuration: " + str(config))

        elif (line[0:18] == 'Minimum intensity:'):
            l = line.split()
            min_intensity = l[2]
            print("min_intensity: " + min_intensity)

        elif (line[0:18] == 'Maximum intensity:'):
            l = line.split()
            max_intensity = l[2]
            print("max_intensity: " + max_intensity)

        elif (line[0:11] == 'Intensity :'):
            l = line.split()
            intensityList.append(l[2])
            
        elif (line[0:15] == 'Correct answer:'):
            l = line.split()
            l = str(l[2])
            l = np.array(l)
            correctList.append(l)

        elif (line[0:17] == 'Patient\'s answer:'):
            l = line.split()
            l = str(l[2])
            l = np.array(l)
            patientList.append(l)

    intensityList = np.array(intensityList).astype('float64')
    correctList = np.array(correctList).astype('float64')
    patientList = np.array(patientList).astype('float64')
    
    scoreCounter = 0
    totalCounter = 0
    for i in range(len(patientList)):
        if correctList[i] == patientList[i]:
            scoreCounter = scoreCounter + 1
        totalCounter = totalCounter + 1
        

    print("Intensities: " + str(intensityList))
    print("correctList: " + str(correctList))
    print("patientList: " + str(patientList))
    print(' ')
    print("Accuracy: " + str((scoreCounter / totalCounter)* 100 ))

    #fig = plt.figure(figsize = (15,5))
    #plt.plot(intensityList,correctList, color = "blue", marker='o', linewidth = 2)
    #plt.ylim((0, 100))
    #plt.title('Accuracy(%) vs Intensity(uA), electrode ' + str(electrode))

    return electrode,trials,config,intensityList

def analyze_2AFC_new(f,y_axis_min,show_prints):

    electrodes = []
    intensitiesList = []
    pati_responseList = []
    corr_responseList = []
    score_list = []
    timingsList = []

    Stimulus_type = '' #This will be either single train or blinking


    for line in f:


        if (line[0:19] == 'Configuration Tags:'):
            configuration_tags = line[20:-1]
            print('config tags ' + str(configuration_tags) + '\n') 

        elif (line[0:14] == 'Configuration:'):

            config = line[15:-1]
            print('config ' + str(config) + '\n') 

        elif (line[0:10] == 'Electrode:'):

            electrodes = line[11:-1]
            #electrodes = electrodes[1:-1]
            print('electrodes ' + str(electrodes) + '\n') 

        elif (line[0:15] == 'Single Stimulus'):

            Stimulus_type = 'Single Stimulus'
            print('Stimulus_type ' + Stimulus_type + '\n') 

        elif (line[0:8] == 'Blinking'):

            Stimulus_type = 'Blinking with repetitions=10 and train frequency=3Hz'
            print('Stimulus_type' + Stimulus_type + '\n') 

        elif (line[0:3] == 'Is:'):

            intensities = line[3:-1]

            a = intensities.split(",")

            for j in range(len(a)):

                if j == 0:
                    a_aux = a[j]
                    a_aux = a_aux.split('[')
                    a_aux = a_aux[1]
                    intensitiesList.append(a_aux)  

                elif j == len(a)-1:
                    a_aux = a[j]
                    a_aux = a_aux.split(']')
                    a_aux = a_aux[0]
                    intensitiesList.append(a_aux)  

                else:
                    intensitiesList.append(a[j])


        elif (line[0:3] == 'Rs:'):

            responses = line[3:-1]

            a = responses.split(",")

            for j in range(len(a)):

                if j == 0:
                    a_aux = a[j]
                    a_aux = a_aux.split('[')
                    a_aux = a_aux[1]
                    pati_responseList.append(a_aux)  

                elif j == len(a)-1:
                    a_aux = a[j]
                    a_aux = a_aux.split(']')
                    a_aux = a_aux[0]
                    pati_responseList.append(a_aux)  

                else:
                    pati_responseList.append(a[j])

        elif (line[0:4] == 'CRs:'):

            responses = line[3:-1]

            a = responses.split(",")

            for j in range(len(a)):

                if j == 0:
                    a_aux = a[j]
                    a_aux = a_aux.split('[')
                    a_aux = a_aux[1]
                    corr_responseList.append(a_aux)  

                elif j == len(a)-1:
                    a_aux = a[j]
                    a_aux = a_aux.split(']')
                    a_aux = a_aux[0]
                    corr_responseList.append(a_aux)  

                else:
                    corr_responseList.append(a[j])

                    
        elif (line[0:3] == 'Ds:'):

            timings = line[3:-1]

            a = timings.split(",")
            #print("timings : " + str(a) +'\n') 

            for j in range(len(a)):

                if j == 0:
                    a_aux = a[j]
                    a_aux = a_aux.split('[')
                    a_aux = str(a_aux[1])
                    #print("a_aux : l" + str(a_aux) +'\n') 
                    timingsList.append(a_aux[1:-1])
                    
                elif j == len(a)-1:
                    a_aux = a[j]
                    a_aux = a_aux.split(']')
                    #print("a_aux : " + str(a_aux) +'\n') 
                    a_aux = a_aux[0:]
                    timingsList.append(a_aux)  

                else:
                    #print("a : l" + str(a[j]) +'\n') 
                    a_aux = str(a[j])
                    timingsList.append(a_aux[1:-1])    
                    #print("a : l" + str(a_aux[1:-1]) +'\n') 


    intensitiesList = np.array(intensitiesList).astype('uint8')
    pati_responseList = np.array(pati_responseList).astype('uint8')
    corr_responseList = np.array(corr_responseList).astype('uint8')
    score_list = pati_responseList == corr_responseList
    aux_scores = []
    
    for i in range(len(score_list)):
        if score_list[i] == True:
            aux_scores.append(1)
        else:
            aux_scores.append(0)  
            
    score_list = np.array(aux_scores)
    #scoresList = np.array(scoresList).astype('uint8')
    #timingsList = np.array(timingsList).astype('uint8')
    #print('intensitiesList ' + str(intensitiesList))
    #print('pati_responseList ' + str(pati_responseList))
    #print('corr_responseList ' + str(corr_responseList))
    #print('score_list ' + str(score_list))


    #finding unique values
    uni = np.unique(intensitiesList)
    accumulated_scores = []
    accumulated_timings = []

    for i in range(len(uni)):
        a = np.where(intensitiesList  == uni[i])
        suma_scores = np.sum(score_list[a])
        accumulated_scores.append(suma_scores)

    num_trials = len(intensitiesList) / np.squeeze(np.array([len(uni)]).astype('uint8'))

    accumulated_scores = np.array(accumulated_scores).astype('uint8')
    percentage_scores = (accumulated_scores/float(num_trials)*100)

    accumulated_timings = np.array(accumulated_timings).astype('uint8')
    mean_timings = (accumulated_timings/float(num_trials))
    
    if show_prints == 1:
        print("Total Number of Trials : " + str(len(intensitiesList)) +'\n') 
        print("Configuration: " + str(config)+'\n') 
        print("intensities  " + str(uni)+'\n') 
        print("acc scores   " + str(accumulated_scores) +'\n') 
        print("percentage scores   " + str(percentage_scores) +'\n') 
        #print("mean timings   " + str(mean_timings) +'\n') 
        print("num trials   " + str(num_trials) +'\n') 
        print(" ")

    fig = plt.figure(figsize = (10,3))
    plt.plot(uni,percentage_scores, color = "blue", marker='o', linewidth = 2)
    plt.ylim((y_axis_min, 100))
    plt.title('Perceptions(%) vs Intensity(uA), electrode ' + str(electrodes) + ' ' + str(config))

    return uni, percentage_scores