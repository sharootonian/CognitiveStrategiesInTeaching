
  // Display alert message on back/refresh.
  // https://developer.mozilla.org/en-US/docs/Web/API/WindowEventHandlers/onbeforeunload
  function verify_unload(e){
    e.preventDefault();
    (e || window.event).returnValue = null;
    return null;
  };
  window.addEventListener("beforeunload", verify_unload);

  var jsPsych = initJsPsych({
    show_progress_bar: true,
    auto_update_progress_bar: false,
    on_finish: function() {
      // jsPsych.data.displayData();
      // Remove requirement to verify redirect
      window.removeEventListener("beforeunload", verify_unload);

      // Add interactions to the data variable
      var interaction_data = jsPsych.data.getInteractionData();
      jsPsych.data.get().addToLast({interactions: interaction_data.json()});

      // Display jsPsych data in viewport.
      // Dump data to JSON.
      var json = jsPsych.data.get().json();

      // Display data on screen for demo purposes
      jsPsych.data.displayData(); 

    }
  })
    
    // initializes timeline (controls which order and how long they'll stay up per trial)
    var timeline = [];
    var current_index= []
    var flipped_trial = []
    var current_congruency = []
    var id = []
    var quiz_fail = 0

    // generate a random subject ID with 15 characters
    var subject_id = jsPsych.randomization.randomID(15);
    var low_quality = false
    // Define experiment fullscreen.
    var enter_fullscreen = {
      type: jsPsychFullscreen,
      fullscreen_mode: true
    }

    // Demo mode - no worker IDs needed

    // // pick a random condition for the subject at the start of the experiment
    // var condition_assignment = jsPsych.randomization.sampleWithoutReplacement(['conditionA', 'conditionB', 'conditionC'], 1)[0];

    // PRELOADS NECESSARY IMAGES
    var preload = {
        type: jsPsychPreload,
        images: [
        '../../shared/img/inst_0.png',
        '../../shared/img/inst_0_labeled.png',
        '../../shared/img/inst_1.png',
        '../../shared/img/inst_2.png',
        '../../shared/img/inst_3.png',
        '../../shared/img/inst_3_advice_teacher.png',
        '../../shared/img/inst_3_teacher.png',
        '../../shared/img/inst_advice_0.png',
        '../../shared/img/inst_advice_1.png',
        '../../shared/img/inst_advice_2.png',
        '../../shared/img/inst_advice_3.png',
        '../../shared/img/Inst_badadvice_0.png',
        '../../shared/img/Inst_badadvice_1.png',
        '../../shared/img/Inst_badadvice_2.png',
        '../../shared/img/Inst_badadvice_3.png',
        '../../shared/img/Inst_badadvice_teacher.png',
        '../../shared/img/training_1_correctanswer.png',
        '../../shared/img/training_2_correctanswer.png',
        '../../shared/img/training_3_correctanswer1.png',
        '../../shared/img/training_3_correctanswer2.png',
      ]
      };
    timeline.push(preload);

    // welcome screen trial
    var welcome = {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: "Welcome to the experiment. Press any key to begin.",
        on_start: function() {
            // set progress bar to 0 at the start of experiment
            jsPsych.setProgressBar(0);
        }
      };
    //adds to end of timeline array

    // instruction trial
    var instructions = {
      type: jsPsychInstructions,
        pages: [
        //page 1
        '<h1 style="text-align:center">We are now starting the experiment. </h1> <h3>Use the buttons below (or the left/right arrow keys) to navigate the instructions.</h3>',
        //page 2
        '<h3>You are going to be playing a teaching game by helping a student take the best path.</h3> '+
        '<h3>First, let\'s look step by step at how a student decides on a path.</h3>',
        //page 3
        '<p style="text-align: left; width: 800px; border: 1px solid white; margin: 0 auto;">\
          The student will be navigating down a graph (shown below) gathering points along the way. \
          <br>\<br>\
          They will always start at the top circle and move downwards to one of the circles at the end. Each circle gives some # of points and the dashed lines are possible paths the student can take. \
          <br>\<br>\
          Note: the student can never move upwards and must always go down or diagonally down.</p>'+
        '<img src="../../shared/img/inst_0_labeled.png" width="350"> </img>'+'<p>let\'s look step by step at how the student navigates with this new infromation.</p>'+'<p>press next</p>', 
        //page 4
        '<img src="../../shared/img/inst_0.png" width="350"></img>'+'<p style="color:white;">.</p>'+'<p style="color:white;"> . <br>\.</p>'+'<p>press next</p>',// (some fluff to keep thing aligned across pages)
        //page 5
        '<img src="../../shared/img/inst_1.png" width="350"></img>'+'<p style="color:white;">.</p>'+'<p style="color:white;"> . <br>\.</p>'+'<p>press next</p>',
        //page 6
        '<img src="../../shared/img/inst_2.png" width="350"></img>'+'<p style="color:white;">.</p>'+'<p style="color:white;"> . <br>\.</p>'+'<p>press next</p>',
        //page 7
        '<img src="../../shared/img/inst_3.png" width="350"></img>'+'<p>The student is done navigating and earned <b>4 points</b>.</p>'+'<p style="color:white;"> . <br>\.</p>'+'<p>press next</p>',
        //page 8
        '<img src="../../shared/img/inst_3.png" width="350"></img>'+ 
        '<p>The student is done navigating and earned <b>4 points</b>.</p>'+
        '<p style="text-align: left; width: 800px; border: 1px solid white; margin: 0 auto;">\
          <b>Note:</b> The student will always take the sequence of paths they think will give the most total points. The student didn\’t know about a path to the circle with the +3 points, so that\’s why they didn\’t go there.</p>'+ '<p>press next</p>',
        //page 9 teacher
        '<p style="text-align: left; width: 800px; border: 1px solid white; margin: 0 auto;">\
          Below is how you, as the <b>teacher</b>, will see the graph. <u>You know all the possible paths.</u> You are able to see what paths the student navigated and how many points they received. <br><br>\
          <u>You don\’t know what other paths the student knows or doesn\'t know, but you can assume that the student took the best path, given what they knew.</u></p>'+
          '<img src="../../shared/img/inst_3_teacher.png" width="350"></img>',
        //page 10
        '<p style="text-align: left; width: 800px; border: 1px solid white; margin: 0 auto;">\
          Now you are going to reveal a single path to the student,<b>to improve the student\'s points the next time they navigate. </b>\
          <br>\<br>\
          For example, let\'s say you chose to reveal the yellow highlighted path to the student</p>'+
        '<img src="../../shared/img/inst_3_advice_teacher.png" width="350"></img>'+
        '<p>let\'s look step by step at how the student navigates with this new infromation. </p>',
        //page 11
        '<img src="../../shared/img/inst_advice_0.png" width="350"></img>'+'<p style="color:white;">.</p>'+'<p style="color:white;"> . </p>'+'<p>press next</p>',// (some fluff to keep thing aligned across pages)
        //page 12
        '<img src="../../shared/img/inst_advice_1.png" width="350"></img>'+'<p style="color:white;">.</p>'+'<p style="color:white;"> . </p>'+'<p>press next</p>',// (some fluff to keep thing aligned across pages)
        //page 13
        '<img src="../../shared/img/inst_advice_2.png" width="350"></img>'+'<p style="color:white;">.</p>'+'<p style="color:white;"> . </p>'+'<p>press next</p>',// (some fluff to keep thing aligned across pages)
        //page 14
        '<img src="../../shared/img/inst_advice_3.png" width="350"></img>'+
        '<p>The student is done navigating and earned <b>6 points</b>.</p>'+
        '<p style="color:white;">.</p>'+
        '<p>press next</p>',
        //page 15
        '<img src="../../shared/img/inst_advice_3.png" width="350"></img>'+
        '<p>The student is done navigating and earned <b>6 points</b>.</p>'+
        '<p>The path you chose to reveal was a good choice as it increased the student\'s points from 4 to 6! </p>'+
        '<p>press next</p>',
        // page 16
        '<p style="text-align: left; width: 800px; border: 1px solid white; margin: 0 auto;">\
          Now let\'s imagine you choose to reveal a different path to the student (highlighted in yellow).</p>'+
        '<img src="../../shared/img/Inst_badadvice_teacher.png" width="350"></img>'+
        '<p>let\'s look step by step at how the student navigates with this new infromation. </p>',
        //page 17
        '<img src="../../shared/img/Inst_badadvice_0.png" width="350"></img>'+'<p style="color:white;">.</p>'+'<p style="color:white;"> . </p>'+'<p>press next</p>',// (some fluff to keep thing aligned across pages)
        //page 18
        '<img src="../../shared/img/Inst_badadvice_1.png" width="350"></img>'+'<p style="color:white;">.</p>'+'<p style="color:white;"> . </p>'+'<p>press next</p>',// (some fluff to keep thing aligned across pages)
        //page 19
        '<img src="../../shared/img/Inst_badadvice_2.png" width="350"></img>'+'<p style="color:white;">.</p>'+'<p style="color:white;"> . </p>'+'<p>press next</p>',// (some fluff to keep thing aligned across pages)
        //page 20
        '<img src="../../shared/img/Inst_badadvice_3.png" width="350"></img>'+
        '<p>The student is done navigating and earned <b>4 points</b>.</p>'+
        '<p>The path we chose to reveal was a <b>bad</b> choice as it didn\'t increase the student\'s points. </p>'+
        '<p>press next</p>',
        //page 21
        '<h3>Hopefully, you are getting the hang of it. Let\'s have you actually practice on a few students.</h3>'+
        '<h3>Note: Each of the students you help will know different possible paths. So each student is unique.</h3>',
        //page 22
        '<h1>We are going to start the training now.</h1>'+
        '<h1>You will not be able to go back to see the examples.</h1>'+
        '<h3>Please make sure you understand the instructions and examples before moving forward. </h3>'
        ],
        show_clickable_nav: true,
        on_finish: function() {
              // at the end of each trial, update the progress bar
              // based on the current value and the proportion to update for each trial
              var curr_progress_bar_value = jsPsych.getProgressBarCompleted();
              jsPsych.setProgressBar(curr_progress_bar_value + (5/(ntrials+20)));
          }
        //post_trial_gap: 2000 // two second gap after previous trial
      };

    // HERE IS WHERE THE TRAINING BEGINS
    var trainingInteractive = {
        type: jsPsychExternalHtml,
        // url: "./nodeGraphs.html",
        url: "./nodeGraphsTraining.html",
        cont_btn: "next_trial",
        check_fn: response_received_training,
        edge_res: getEdgeRes,
        currEdgeRTMS: getCurrEdgeRTMS,
        skipped: getSkipped,
        flipped: getFlipped,
        current_correct_edge: getCurrentEdge,
        current_index: getCurrentIndex,
        seed : getModelSeed,
        congruency: getCongruency,
        trial_id: getTrial_id,
        responsesLog: getResponsesLog,
        force_refresh: true,//forces it not to use saved cookies
        execute_script: true,// executes JS scripts on external page
        on_finish: function() {
            // at the end of each trial, update the progress bar
            // based on the current value and the proportion to update for each trial
            var curr_progress_bar_value = jsPsych.getProgressBarCompleted();
            jsPsych.setProgressBar(curr_progress_bar_value + (5/(ntrials+20)));
        }
    };

    //testing a quiz
    var quiz = {
        type: jsPsychSurveyMultiChoice,
        preamble: '<h2>Before starting the experiment please complete the following comprehension questions.</h2>',
        questions: [
          {
            prompt: "What is your goal in this task?", 
            name: 'q1', 
            options: ['The goal is for you, as the teacher, to get the most points.', 'The goal is to help the student to get the most points','The goal is to help the student find the quickest path.'], 
            correct: 'The goal is to help the student to get the most points',
            required: true,
            horizontal: false
          }, 
          {
            prompt: "There is always one best path to reveal?", 
            name: 'q2', 
            options: ['True', 'False'], 
            required: true,
            horizontal: false
          },
          {
            prompt: "What do the dotted lines represent when selecting a path?", 
            name: 'q3', 
            options: ['Possible paths you, as the teacher, can choose to teach the learner.', 'Possible paths you, as the teacher, can navigate.','Paths that the student took before.'], 
            required: true,
            horizontal: false
          },
          {
            prompt: "What do the solid black lines represent when selecting a path?", 
            name: 'q4', 
            options: ['Possible paths you, as the teacher, can choose to teach the learner.', 'Possible paths you, as the teacher, can navigate.','Paths that the student took before.'], 
            required: true,
            horizontal: false
          }
        ],
        on_finish: function() {
            // at the end of each trial, update the progress bar
            // based on the current value and the proportion to update for each trial
            var curr_progress_bar_value = jsPsych.getProgressBarCompleted();
            jsPsych.setProgressBar(curr_progress_bar_value + (5/(ntrials+20)));

            var incorrect = false;
            if (jsPsych.data.getLastTrialData().values()[0].response.q1 != 'The goal is to help the student to get the most points') {
              incorrect = true;
            }
            if (jsPsych.data.getLastTrialData().values()[0].response.q2 != 'False') {
              incorrect = true;
            }
            if (jsPsych.data.getLastTrialData().values()[0].response.q3 != 'Possible paths you, as the teacher, can choose to teach the learner.') {
              incorrect = true;
            }
            if (jsPsych.data.getLastTrialData().values()[0].response.q4 != 'Paths that the student took before.') {
              incorrect = true;
            }
            if (incorrect == true) {
            quiz_fail=1
            }
          }
        } 

    //testing a quiz
    var last_quiz = {
        type: jsPsychSurveyMultiChoice,
        preamble: '<h1>Incorrect<h1>\
        <h2>You got one or more questions wrong. <u>This is your last chance to answer the following questions correctly.</u></h2>',
        questions: [
          {
            prompt: "What is your goal in this task?", 
            name: 'q1', 
            options: ['The goal is for you, as the teacher, to get the most points.', 'The goal is to help the student to get the most points','The goal is to help the student find the quickest path.'], 
            correct: 'The goal is to help the student to get the most points',
            required: true,
            horizontal: false
          }, 
          {
            prompt: "There is always one best path to reveal?", 
            name: 'q2', 
            options: ['True', 'False'], 
            required: true,
            horizontal: false
          },
          {
            prompt: "What do the dotted lines represent when selecting a path?", 
            name: 'q3', 
            options: ['Possible paths you, as the teacher, can choose to teach the learner.', 'Possible paths you, as the teacher, can navigate.','Paths that the student took before.'], 
            required: true,
            horizontal: false
          },
          {
            prompt: "What do the solid black lines represent when selecting a path?", 
            name: 'q4', 
            options: ['Possible paths you, as the teacher, can choose to teach the learner.', 'Possible paths you, as the teacher, can navigate.','Paths that the student took before.'], 
            required: true,
            horizontal: false
          }
        ],
        on_finish: function() {
            // at the end of each trial, update the progress bar
            // based on the current value and the proportion to update for each trial
            var curr_progress_bar_value = jsPsych.getProgressBarCompleted();
            jsPsych.setProgressBar(curr_progress_bar_value + (5/(ntrials+20)));

            var incorrect = false;
            if (jsPsych.data.getLastTrialData().values()[0].response.q1 != 'The goal is to help the student to get the most points') {
              incorrect = true;
            }
            if (jsPsych.data.getLastTrialData().values()[0].response.q2 != 'False') {
              incorrect = true;
            }
            if (jsPsych.data.getLastTrialData().values()[0].response.q3 != 'Possible paths you, as the teacher, can choose to teach the learner.') {
              incorrect = true;
            }
            if (jsPsych.data.getLastTrialData().values()[0].response.q4 != 'Paths that the student took before.') {
              incorrect = true;
            }
            if (incorrect == true) {
            // In demo mode, don't end experiment - let them continue
            }
          }
        } 

        

    function getQuizFail(){
      return quiz_fail;
    }

    var quiz_failed = {
    timeline: [last_quiz],
    conditional_function: function(){
        if (getQuizFail() == 1) {
          return true
        } else{
          return false
        }

      }
    }


    //take a break if they want
    var breaktime = {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: '<p>You can take a 60 second break before starting the experiment.</p><br><br><p>This page will advance in 60 seconds or you can press any key to advance now.</p>',
        trial_duration: 60000,
      };
    
    // HERE IS WHERE THE TRAINING ENDS, AND THE ACTUAL TASK BEGINS
    var experiment_reminder = {
        type: jsPsychInstructions,
        pages: [
        '<p style="text-align: left; width: 800px; border: 1px solid white; margin: 0 auto;">\
          This is a demo version of the experiment.\
          <br>\<br>\
          <u>Reminders:</u> <br>\
          Every student has a different set of possible paths they can take. <br>\
          Every student tries their best, as in they will always take the paths that will yield the most total points given their knowledge.\
          <br>\<br>\</p>'+
          '<p>Click Next to start the Experiment.</p>'
        ],
        show_clickable_nav: true
    }

      var edgeSelection = [];
      var responsesLog = [];
      var currEdgeRTMS = undefined;
      var skipped = [];
      // function that checks if a selection has been made
      var response_received = function(elem) {
          if (document.getElementById("selected").checked) {
            edgeSelection.push(sessionStorage.getItem("edgeSelection"))
            sessionStorage.setItem("edgeSelection","")

            responsesLog = (sessionStorage.getItem("responsesLog"))
            currEdgeRTMS = (sessionStorage.getItem("edgeRTMS"))
            skipped = (sessionStorage.getItem("skipped"))
            return true;
          }
          else {
            alert("you must select a path first.");
            return false;
          }
          return false;
      };

      var response_received_training = function(elem) {
          if (document.getElementById("selected").checked) {
            sessionStorage.setItem("edgeSelection","")

            return true;
          }
          else {
            alert("you must finish all training first.")
            return false;
          }
          return false;
      }

    function getCurrentIndex() {
      current_index=JSON.parse(localStorage.getItem("shuffledIndices"))[sessionStorage.indexInput-1]
      return current_index
    }
    function getCurrentEdge() {
      // Exp2 doesn't use correct edges, return null
      return null;
    }
    function getFlipped() {
      flipped_trial=JSON.parse(localStorage.getItem("flipped"))[sessionStorage.indexInput-1]
      return flipped_trial
    }
    function getModelSeed() {
      // Exp2 doesn't use model seeds, return null
      return null;
    }
    function getTrial_id() {
      id=JSON.parse(localStorage.getItem("trial_id"))[sessionStorage.indexInput-1]
      return id
    }

    function getCongruency() {
      current_congruency =JSON.parse(localStorage.getItem("congruency"))[sessionStorage.indexInput-1]
      return current_congruency
    }
    
    function getEdgeRes(){
      return edgeSelection[edgeSelection.length - 1];
    }

    function getResponsesLog(){
      return responsesLog;
    }
    function getCurrEdgeRTMS(){
      return currEdgeRTMS;
    }
    function getSkipped(){
      return skipped;
    }

    
    // declare the block.
    var experiment = {
        type: jsPsychExternalHtml,
        // url: "./nodeGraphs.html",
        url: "./nodeGraphs.html",
        cont_btn: "next_trial",
        check_fn: response_received,//waits for true to be returned before it continues
        edge_res: getEdgeRes,
        currEdgeRTMS: getCurrEdgeRTMS,
        skipped: getSkipped,
        flipped: getFlipped,
        current_correct_edge: getCurrentEdge,
        current_index: getCurrentIndex,
        seed : getModelSeed,
        congruency: getCongruency,
        responsesLog: getResponsesLog,
        trial_id: getTrial_id,
        force_refresh: true,//forces it not to use saved cookies
        execute_script: true,// executes JS scripts on external page
        data: {//lets us mark the data
            task: 'experiment',
            subjectId: subject_id
        },
        on_finish: function() {
            // at the end of each trial, update the progress bar
            // based on the current value and the proportion to update for each trial
            var curr_progress_bar_value = jsPsych.getProgressBarCompleted();
            jsPsych.setProgressBar(curr_progress_bar_value + (1/(ntrials+20)));
        }
    };
    var ntrials = 25
    // add graph trials to timeline
    var experimentF_procedure = {
        timeline: [experiment],
        // timeline_variables: test_stimuli,
        post_trial_gap: 500,
        randomize_order: false,
        repetitions: ntrials // THE NUMBER OF ROWS IN THE CSV (aka number of trials we will have)
      };

    var debrief = {
      type: jsPsychSurvey,
      pages: [
        [
          {
            type: 'html',
            prompt: '<h3>This is a demo - no data will be collected.</h3>',
          },
          {
            type: 'likert',
            prompt: 'How mentally demanding was the task?',
            likert_scale_min_label: 'Not at all',
            likert_scale_max_label: 'Very much',
            likert_scale_values: [
              {value: 1},
              {value: 2},
              {value: 3},
              {value: 4},
              {value: 5},
              {value: 6},
              {value: 7}
              
            ]
          }, 
          {
            type: 'likert',
            prompt: 'How clear were the task instructions?',
            likert_scale_min_label: 'Not at all',
            likert_scale_max_label: 'Very much',
            likert_scale_values: [
              {value: 1},
              {value: 2},
              {value: 3},
              {value: 4},
              {value: 5},
              {value: 6},
              {value: 7}
              
            ]
          }, 
          {
            type: 'likert',
            prompt: 'How successful were you in accomplishing what you were asked to do during the task?',
            likert_scale_min_label: 'Not at all',
            likert_scale_max_label: 'Very much',
            likert_scale_values: [
              {value: 1},
              {value: 2},
              {value: 3},
              {value: 4},
              {value: 5},
              {value: 6},
              {value: 7}
              
            ]
          }, 
          {
            type: 'likert',
            prompt: 'How hard did you have to work to accomplish your level of performance?',
            likert_scale_min_label: 'Not at all',
            likert_scale_max_label: 'Very much',
            likert_scale_values: [
              {value: 1},
              {value: 2},
              {value: 3},
              {value: 4},
              {value: 5},
              {value: 6},
              {value: 7}
              
            ]
          }, 
          {
            type: 'likert',
            prompt: 'How discouraged, irritated, stressed, or annoyed were you during the task?',
            likert_scale_min_label: 'Not at all',
            likert_scale_max_label: 'Very much',
            likert_scale_values: [
              {value: 1},
              {value: 2},
              {value: 3},
              {value: 4},
              {value: 5},
              {value: 6},
              {value: 7}
              
            ]
          }, 
          {
            type: 'text',
            prompt: "What strategy did you use when choosing a path?", 
            name: 'strategies', 
            required: false,
          }, 
          {
            type: 'text',
            prompt: "Do you have any other comments or feedback?", 
            name: 'feedback', 
            required: false,
          },   
        ]
      ],
      title: 'Debriefing',
      button_label_next: 'Continue',
      button_label_back: 'Previous',
      button_label_finish: 'Submit',
      show_question_numbers: 'onPage',
      //compute bonus
      on_finish: function() {
          // at the end of each trial, update the progress bar
          // based on the current value and the proportion to update for each trial
          var curr_progress_bar_value = jsPsych.getProgressBarCompleted();
          jsPsych.setProgressBar(curr_progress_bar_value + (5/(ntrials+20)));
          }
    };

    var edges_selected=[];

    function getStimulusStr() {
      var stimulusStr = '<p>Thanks for participating in this demo!</p>\
        <p>Press any key to see your data.</p>'
      return stimulusStr
    }

    var conclusion = {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: function() {  
          // this question prompt is dynamic - the text that is shown 
          // will change based on the participant's earlier response
          var text = getStimulusStr()
          return text;
        }, 
        // post_trial_gap: 2000, //two second gap after previous trial
        data: {//lets us mark the data
            task: 'end',
            subjectId: subject_id
        },
      };
