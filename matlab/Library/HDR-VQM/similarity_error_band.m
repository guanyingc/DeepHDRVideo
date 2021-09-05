function sim =similarity_dc(v1,v2) 
C2 = 0.2; %small fixed constant
sim_vector=(2*v1.*v2+C2)./(v1.^2+v2.^2+C2);
sim = (sim_vector);


