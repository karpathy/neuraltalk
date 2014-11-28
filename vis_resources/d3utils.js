
function plotToDiv(div, ydata, xdata, params) {
  params = params || {};
  
  var dw = div.offsetWidth;
  var dh = div.offsetHeight;

  var m = [40, 0, 40, 100]; // margins
  var w = dw - m[1] - m[3]; // width
  var h = dh - m[0] - m[2]; // height

  var xmax = 'xmax' in params ? params.xmax : d3.max(xdata);
  var ymax = 'ymax' in params ? params.ymax : d3.max(ydata);
  
  var x = d3.scale.linear().domain([0, xmax]).range([0, w]);
  var y = d3.scale.linear().domain([0, ymax]).range([h, 0]);

  // create a line function that can convert data[] into x and y points
  var line = d3.svg.line()
    .x(function(d,i) { 
      return x(xdata[i]); 
    })
    .y(function(d,i) { 
      return y(ydata[i]); 
    })

    // Add an SVG element with the desired dimensions and margin.
    var graph = d3.select(div).append("svg:svg")
          .attr("width", w + m[1] + m[3])
          .attr("height", h + m[0] + m[2])
        .append("svg:g")
          .attr("transform", "translate(" + m[3] + "," + m[0] + ")");

    var xAxis = d3.svg.axis().scale(x).tickSize(-h).tickSubdivide(true);
    graph.append("svg:g")
          .attr("class", "x axis")
          .attr("transform", "translate(0," + h + ")")
          .call(xAxis);

    var yAxisLeft = d3.svg.axis().scale(y).ticks(4).orient("left");
    graph.append("svg:g")
          .attr("class", "y axis")
          .attr("transform", "translate(-25,0)")
          .call(yAxisLeft);
    
    graph.append("svg:path").attr("d", line(xdata));
    
}