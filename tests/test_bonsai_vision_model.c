/* Real-model Bonsai vision/session regression. Usage:
 * test_bonsai_vision_model MODEL MMPROJ IMAGE [--lifecycle]
 * Uses at most 64 image tokens to keep four supported file combinations cheap.
 * Optional lifecycle covers image identities, validation, reset and replay.
 * Run one model process at a time. No approximate cross-format logit threshold.
 */
#include "../ds4.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char error[512];
static void need(bool ok,const char *message) {
    if(!ok) {fprintf(stderr,"FAIL %s: %s\n",message,error);exit(1);}
}
static void copy(ds4_session *s,float *out,int vocab) {
    need(ds4_session_copy_logits(s,out,vocab)==vocab,"complete vocabulary");
    for(int i=0;i<vocab;++i)need(isfinite(out[i]),"finite logits");
}
static void sync_image(ds4_session *s,const ds4_tokens *prompt,const ds4_vision_span *span) {
    need(ds4_session_sync_multimodal(s,prompt,span,1,error,sizeof(error))==0,"multimodal sync");
    need(ds4_session_pos(s)==prompt->len,"causal position");
    need(ds4_session_vision_state_matches(s,span,1),"stored image identity");
}
int main(int argc,char **argv) {
    need(argc==4 || (argc==5&&!strcmp(argv[4],"--lifecycle")),"usage MODEL MMPROJ IMAGE [--lifecycle]");
    need(setenv("DS4_QWEN4_IMAGE_MAX_TOKENS","64",1)==0,"bounded image test");
    const bool lifecycle=argc==5;
    ds4_engine_options options={.model_path=argv[1],.vision_path=argv[2],
        .backend=DS4_BACKEND_METAL,.context_size=1024,.prefill_chunk=128,.n_threads=4};
    ds4_engine *e=NULL;ds4_session *s=NULL;
    need(ds4_engine_open(&e,&options)==0,"open Bonsai plus mmproj");
    need(ds4_engine_is_bonsai(e)&&ds4_engine_has_vision(e),"Bonsai vision advertised");
    ds4_vision_embedding embedding={0};ds4_vision_span span={0};
    need(ds4_engine_vision_encode_file(e,argv[3],&embedding,error,sizeof(error)),"encode image");
    need(embedding.token_count==(uint64_t)embedding.grid_width*embedding.grid_height,"image grid");
    printf("IMAGE tokens=%u grid=%ux%u\n",embedding.token_count,embedding.grid_height,embedding.grid_width);
    const int embd=ds4_engine_embd_dim(e),vocab=ds4_engine_vocab_size(e);
    for(size_t i=0;i<(size_t)embedding.token_count*embd;++i)need(isfinite(embedding.data[i]),"finite image embeddings");
    ds4_tokens prompt={0},extended={0};ds4_chat_begin(e,&prompt);
    const char *parts[]={"Read exactly the text written in this image. ",""};
    need(ds4_chat_append_multimodal_message(e,&prompt,"user",parts,&embedding,1,&span,error,sizeof(error)),"format Qwen image chat");
    ds4_chat_append_assistant_prefix(e,&prompt,DS4_THINK_NONE);
    need(ds4_session_create(&s,e,1024)==0,"create multimodal session");
    float *front=malloc((size_t)vocab*sizeof(float)),*actual=malloc((size_t)vocab*sizeof(float));
    need(front&&actual,"logit buffers");
    sync_image(s,&prompt,&span);copy(s,front,vocab);
    sync_image(s,&prompt,&span);copy(s,actual,vocab);
    need(!memcmp(front,actual,(size_t)vocab*sizeof(float)),"same image prefix bit-exact");
    ds4_tokens_copy(&extended,&prompt);
    printf("TEXT ");
    for(unsigned step=0;step<12;++step) {
        const int token=ds4_session_argmax(s);
        size_t len=0;char *piece=ds4_token_text(e,token,&len);
        if(piece){fwrite(piece,1,len,stdout);free(piece);}
        need(ds4_session_eval(s,token,error,sizeof(error))==0,"text decode after image");
        ds4_tokens_push(&extended,token);copy(s,actual,vocab);
    }
    putchar('\n');fflush(stdout);
    sync_image(s,&extended,&span); // Image identity must survive ordinary decode.
    if(lifecycle) {
        copy(s,actual,vocab);
        float *saved=malloc((size_t)vocab*sizeof(float));need(saved!=NULL,"saved logits");
        memcpy(saved,actual,(size_t)vocab*sizeof(float));
        const ds4_tokens bad_prompts[]={
            {.v=NULL,.len=-1}, {.v=NULL,.len=extended.len},
            {.v=extended.v,.len=0}, {.v=extended.v,.len=1024},
            {.v=extended.v,.len=INT_MAX}
        };
        // Neither token nor embedding storage may be inspected before these
        // prompt errors are rejected. The invalid pointer is never dereferenced.
        ds4_vision_span unreadable=span;unreadable.embedding.data=(float *)(uintptr_t)1;
        for(size_t i=0;i<sizeof(bad_prompts)/sizeof(bad_prompts[0]);++i) {
            need(ds4_session_sync_multimodal(s,&bad_prompts[i],&unreadable,1,error,sizeof(error))!=0,"invalid prompt rejected before spans");
            need(ds4_session_pos(s)==extended.len,"invalid prompt preserves checkpoint");
            need(ds4_session_vision_state_matches(s,&span,1),"invalid prompt preserves image identity");
            copy(s,actual,vocab);need(!memcmp(saved,actual,(size_t)vocab*sizeof(float)),"invalid prompt preserves logits");
        }
        ds4_vision_span invalid=span;invalid.embedding.grid_width++;
        need(ds4_session_sync_multimodal(s,&extended,&invalid,1,error,sizeof(error))!=0,"invalid image grid rejected");
        copy(s,actual,vocab);need(!memcmp(saved,actual,(size_t)vocab*sizeof(float)),"invalid grid preserves logits");
        const float before=span.embedding.data[0];span.embedding.data[0]=NAN;
        need(ds4_session_sync_multimodal(s,&extended,&span,1,error,sizeof(error))!=0,"nonfinite embedding rejected");
        span.embedding.data[0]=before;
        copy(s,actual,vocab);need(!memcmp(saved,actual,(size_t)vocab*sizeof(float)),"invalid data preserves logits");
        need(ds4_session_pos(s)==extended.len,"invalid data preserves checkpoint");
        for(unsigned field=0;field<3;++field) {
            ds4_vision_span changed=span;
            if(field==0)changed.embedding.grid_width++;
            if(field==1)changed.embedding.grid_height++;
            if(field==2)changed.embedding.layout++;
            need(!ds4_session_vision_prefix_matches(s,&changed,1),"image geometry participates in prefix identity");
            need(!ds4_session_vision_state_matches(s,&changed,1),"image geometry participates in state identity");
            changed.token_start++;
            const uint32_t shifted=changed.token_start;
            need(!ds4_session_rebase_vision_state(s,&changed,1),"rebase rejects changed image geometry");
            need(changed.token_start==shifted,"rejected rebase preserves span");
        }
        ds4_vision_span rebased=span;rebased.token_start++;
        need(ds4_session_rebase_vision_state(s,&rebased,1)&&rebased.token_start==span.token_start,"matching geometry rebases");
        // Keep token IDs, count, bytes and fingerprint unchanged, but choose
        // another valid grid. Reuse must now rebuild the same prompt's state.
        ds4_vision_span regridded=span;
        regridded.embedding.grid_width=span.embedding.grid_width==span.embedding.token_count ? 1u : span.embedding.token_count;
        regridded.embedding.grid_height=span.embedding.token_count/regridded.embedding.grid_width;
        sync_image(s,&extended,&regridded);copy(s,actual,vocab);
        need(memcmp(saved,actual,(size_t)vocab*sizeof(float))!=0,"changed grid with same fingerprint recomputes logits");
        memcpy(saved,actual,(size_t)vocab*sizeof(float));
        ds4_session_invalidate(s);sync_image(s,&extended,&regridded);copy(s,actual,vocab);
        need(!memcmp(saved,actual,(size_t)vocab*sizeof(float)),"changed grid matches cold replay");
        error[0]=0;free(saved);
        // Same token IDs with a new image identity must rebuild the recurrent
        // state. Equal embedding bytes make the rebuilt result an exact oracle.
        span.embedding.fingerprint[0]^=0x80;
        sync_image(s,&prompt,&span);copy(s,actual,vocab);
        need(!memcmp(front,actual,(size_t)vocab*sizeof(float)),"changed image identity rebuild");
        ds4_session_invalidate(s);sync_image(s,&prompt,&span);copy(s,actual,vocab);
        need(!memcmp(front,actual,(size_t)vocab*sizeof(float)),"reset restores MRoPE origin");
        ds4_session_rewind(s,0);sync_image(s,&prompt,&span);copy(s,actual,vocab);
        need(!memcmp(front,actual,(size_t)vocab*sizeof(float)),"rewind replay MRoPE");
        puts("PASS image identity, invalid input atomicity, reset and rewind replay");
    }
    puts("PASS Bonsai image encode, multimodal prefill, full vocabulary, image cache and text continuation");
    free(front);free(actual);ds4_session_free(s);ds4_tokens_free(&prompt);ds4_tokens_free(&extended);
    ds4_vision_embedding_free(&span.embedding);ds4_engine_close(e);return 0;
}
